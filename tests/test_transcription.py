"""Tests for core/transcription.py — run with `python -m unittest discover tests`.

Covers the crown-jewel diarization logic (ROADMAP Phase 0), the Deepgram
fallback/retry chain (mocked HTTP), and the sentiment/insights analysis.
No network access is needed.
"""

import os
import tempfile
import unittest
from unittest import mock

from core import transcription as tr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utt(speaker, start, end, text):
    return {"speaker": speaker, "start": start, "end": end, "transcript": text, "words": []}


def make_raw(utterances=None, words=None, transcript=None, duration=None, sentiments=None,
             detected_language=None):
    if transcript is None:
        transcript = " ".join(u["transcript"] for u in (utterances or []))
    channel = {"alternatives": [{"transcript": transcript, "words": words or []}]}
    if detected_language:
        channel["detected_language"] = detected_language
    raw = {
        "metadata": {"duration": duration if duration is not None else
                     (utterances[-1]["end"] if utterances else 0.0)},
        "results": {"channels": [channel]},
    }
    if utterances is not None:
        raw["results"]["utterances"] = utterances
    if sentiments is not None:
        raw["results"]["sentiments"] = sentiments
    return raw


class FakeResponse:
    def __init__(self, status=200, json_data=None, text="err"):
        self.status_code = status
        self._json = json_data
        self.text = text
        self.headers = {}

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


# ---------------------------------------------------------------------------
# format_diarized_output
# ---------------------------------------------------------------------------

class TestFormatDiarizedOutput(unittest.TestCase):
    def test_groups_consecutive_same_speaker(self):
        raw = make_raw([
            utt(0, 0, 2, "Hola."), utt(0, 2, 4, "¿Qué tal?"), utt(1, 4, 6, "Bien."),
        ])
        out = tr.format_diarized_output(raw)
        self.assertEqual(out, "[Speaker 0]: Hola. ¿Qué tal?\n[Speaker 1]: Bien.")

    def test_applies_speaker_mapping(self):
        raw = make_raw([utt(0, 0, 2, "Hola."), utt(1, 2, 4, "Adiós.")])
        out = tr.format_diarized_output(raw, {0: 5, 1: 3})
        self.assertIn("[Speaker 5]: Hola.", out)
        self.assertIn("[Speaker 3]: Adiós.", out)

    def test_falls_back_to_plain_transcript(self):
        raw = make_raw(utterances=None, transcript="solo texto plano")
        self.assertEqual(tr.format_diarized_output(raw), "solo texto plano")

    def test_synthesizes_utterances_from_diarized_words(self):
        words = [
            {"word": "hola", "punctuated_word": "Hola,", "start": 0.0, "end": 0.4, "speaker": 0},
            {"word": "mundo", "punctuated_word": "mundo.", "start": 0.4, "end": 0.8, "speaker": 0},
            {"word": "adiós", "punctuated_word": "Adiós.", "start": 1.0, "end": 1.4, "speaker": 1},
        ]
        raw = make_raw(utterances=None, words=words, transcript="hola mundo adiós")
        out = tr.format_diarized_output(raw)
        self.assertEqual(out, "[Speaker 0]: Hola, mundo.\n[Speaker 1]: Adiós.")

    def test_empty_response(self):
        self.assertEqual(tr.format_diarized_output({}), "")


# ---------------------------------------------------------------------------
# Speaker stats + cross-chunk mapping
# ---------------------------------------------------------------------------

class TestExtractSpeakers(unittest.TestCase):
    def test_stats_have_duration_and_words(self):
        raw = make_raw([
            utt(0, 0, 10, "uno dos tres"), utt(1, 10, 12, "ok"), utt(0, 12, 20, "cuatro cinco"),
        ])
        utterances, stats = tr.extract_speakers_from_response(raw)
        self.assertEqual(len(utterances), 3)
        self.assertAlmostEqual(stats[0]["duration"], 18.0)
        self.assertEqual(stats[0]["count"], 2)
        self.assertEqual(stats[0]["words"], 5)
        self.assertAlmostEqual(stats[1]["duration"], 2.0)


class TestMapSpeakers(unittest.TestCase):
    def test_empty_inputs(self):
        self.assertEqual(tr.map_speakers_between_chunks({}, {0: {"count": 1, "duration": 1, "words": 1}}), {})
        self.assertEqual(tr.map_speakers_between_chunks({0: {"count": 1, "duration": 1, "words": 1}}, {}), {})

    def test_mapping_is_injective_regression(self):
        """Old heuristic could map two current speakers onto ONE previous
        speaker (continuity + rank both pointing at the same target),
        silently merging two voices. Must never happen."""
        prev = {0: {"count": 10, "duration": 100.0, "words": 300},
                1: {"count": 5, "duration": 50.0, "words": 150}}
        cur = {0: {"count": 8, "duration": 80.0, "words": 240},
               1: {"count": 6, "duration": 60.0, "words": 180}}
        mapping = tr.map_speakers_between_chunks(prev, cur, prev_last_speaker=1,
                                                 current_first_speaker=1)
        self.assertEqual(len(set(mapping.values())), len(mapping), f"non-injective: {mapping}")
        # Continuity: speaker who opens chunk N+1 matches speaker who closed chunk N.
        self.assertEqual(mapping[1], 1)
        self.assertEqual(mapping[0], 0)

    def test_matches_by_talk_share(self):
        # Dominant speaker locally labeled 1 in the new chunk must map back to
        # the dominant previous speaker 0.
        prev = {0: {"count": 20, "duration": 200.0, "words": 600},
                1: {"count": 4, "duration": 20.0, "words": 60}}
        cur = {1: {"count": 15, "duration": 180.0, "words": 500},
               0: {"count": 3, "duration": 25.0, "words": 70}}
        mapping = tr.map_speakers_between_chunks(prev, cur)
        self.assertEqual(mapping[1], 0)
        self.assertEqual(mapping[0], 1)

    def test_new_speaker_gets_fresh_id(self):
        prev = {0: {"count": 10, "duration": 100.0, "words": 300},
                1: {"count": 10, "duration": 100.0, "words": 300}}
        cur = {0: {"count": 5, "duration": 50.0, "words": 150},
               1: {"count": 5, "duration": 50.0, "words": 150},
               2: {"count": 5, "duration": 50.0, "words": 150}}
        mapping = tr.map_speakers_between_chunks(prev, cur)
        self.assertEqual(len(set(mapping.values())), 3)
        self.assertIn(2, mapping.values())  # someone got the fresh ID (max prev + 1)

    def test_accepts_legacy_int_counts(self):
        mapping = tr.map_speakers_between_chunks({0: 10, 1: 2}, {0: 9, 1: 3})
        self.assertEqual(mapping, {0: 0, 1: 1})


class TestChunkOrchestration(unittest.TestCase):
    def test_global_ids_consistent_across_chunks(self):
        """Speakers swap local IDs in chunk 2; global labels must not swap."""
        chunk1 = make_raw([
            utt(0, 0, 60, "palabra " * 200), utt(1, 60, 70, "corto"),
        ], duration=70)
        # Same voices, but Deepgram labeled the dominant one "1" this time.
        chunk2 = make_raw([
            utt(1, 0, 55, "palabra " * 180), utt(0, 55, 65, "breve"),
        ], duration=65)

        with mock.patch.object(tr, "transcribe_with_deepgram", side_effect=[chunk1, chunk2]):
            merged = tr._transcribe_chunks(
                ["a.mp3", "b.mp3"], [70.0, 65.0], "key", "es", True, {}, False,
                60, lambda f, m: None,
            )
        transcript = merged["transcript"]
        # Dominant voice is Speaker 0 in BOTH chunks after remapping.
        blocks = transcript.split("\n\n")
        self.assertTrue(blocks[0].startswith("[Speaker 0]:"))
        self.assertTrue(blocks[1].startswith("[Speaker 0]:"))
        self.assertIn("[Speaker 1]: corto", blocks[0])
        self.assertIn("[Speaker 1]: breve", blocks[1])
        # Normalized utterances carry global speakers and absolute offsets.
        self.assertEqual(merged["utterances"][2]["speaker"], 0)
        self.assertAlmostEqual(merged["utterances"][2]["start"], 70.0)

    def test_failed_chunk_reported_not_fatal(self):
        chunk_ok = make_raw([utt(0, 0, 10, "hola " * 30)], duration=10)
        with mock.patch.object(
            tr, "transcribe_with_deepgram",
            side_effect=[{"error": "boom"}, chunk_ok],
        ):
            merged = tr._transcribe_chunks(
                ["a.mp3", "b.mp3"], [10.0, 10.0], "key", "es", True, {}, False,
                60, lambda f, m: None,
            )
        self.assertIn("[Error in chunk 1: boom]", merged["transcript"])
        self.assertTrue(merged["warnings"])
        # Offset of the second chunk still advanced past the failed first one.
        self.assertAlmostEqual(merged["utterances"][0]["start"], 10.0)


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

class TestInsights(unittest.TestCase):
    def _sample(self):
        return [
            utt(0, 0.0, 30.0, "Bueno pues yo creo que o sea el proyecto va bien o sea sí. " + "palabra " * 60),
            utt(1, 30.5, 40.0, "¿Y el presupuesto? " + "dato " * 20),
            utt(0, 39.5, 60.0, "O sea el presupuesto está o sea ajustado. " + "cifra " * 40),
            utt(1, 60.5, 65.0, "Vale."),
        ]

    def test_talk_share_and_words(self):
        insights = tr.compute_speech_insights(self._sample(), language="es")
        s0 = insights["per_speaker"][0]
        s1 = insights["per_speaker"][1]
        self.assertGreater(s0["talk_share"], 0.7)
        self.assertAlmostEqual(s0["talk_share"] + s1["talk_share"], 1.0, places=5)
        self.assertGreater(s0["wpm"], 0)

    def test_interruption_detected(self):
        insights = tr.compute_speech_insights(self._sample(), language="es")
        # Speaker 0 starts at 39.5 while speaker 1 talks until 40.0 -> overlap.
        self.assertEqual(insights["overall"]["interruptions"], 1)
        self.assertEqual(insights["per_speaker"][0]["interruptions_made"], 1)

    def test_fillers_counted_es(self):
        insights = tr.compute_speech_insights(self._sample(), language="es")
        fillers = insights["per_speaker"][0]["fillers"]
        self.assertEqual(fillers.get("o sea"), 4)
        self.assertIn("pues", fillers)

    def test_questions_counted(self):
        insights = tr.compute_speech_insights(self._sample(), language="es")
        self.assertEqual(insights["per_speaker"][1]["questions"], 1)

    def test_dominance_feedback(self):
        insights = tr.compute_speech_insights(self._sample(), language="es")
        self.assertTrue(any("dominó" in f for f in insights["feedback"]))

    def test_monologue_merges_short_gaps(self):
        utterances = [
            utt(0, 0, 70, "a " * 100), utt(0, 71, 140, "b " * 100), utt(1, 141, 150, "ok"),
        ]
        insights = tr.compute_speech_insights(utterances, language="es")
        self.assertGreaterEqual(insights["per_speaker"][0]["longest_monologue"], 140)
        self.assertTrue(any("Monólogo" in f for f in insights["feedback"]))

    def test_english_fillers(self):
        utterances = [utt(0, 0, 20, "Um you know I mean um the plan is um you know fine. " + "word " * 30)]
        insights = tr.compute_speech_insights(utterances, language="en")
        self.assertEqual(insights["per_speaker"][0]["fillers"].get("um"), 3)
        self.assertEqual(insights["per_speaker"][0]["fillers"].get("you know"), 2)

    def test_empty(self):
        self.assertIsNone(tr.compute_speech_insights([]))


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

class TestSentiment(unittest.TestCase):
    def _raw_with_sentiment(self):
        words = []
        for i in range(10):
            words.append({"word": f"w{i}", "start": float(i), "end": i + 0.5,
                          "speaker": 0 if i < 6 else 1})
        sentiments = {
            "segments": [
                {"text": "seg1", "start_word": 0, "end_word": 5,
                 "sentiment": "positive", "sentiment_score": 0.8},
                {"text": "seg2", "start_word": 6, "end_word": 9,
                 "sentiment": "negative", "sentiment_score": -0.6},
            ],
            "average": {"sentiment": "positive", "sentiment_score": 0.24},
        }
        return make_raw(utterances=[], words=words, transcript="x " * 10,
                        duration=10, sentiments=sentiments)

    def test_points_carry_speaker_and_offset(self):
        points = tr.extract_sentiment_points(self._raw_with_sentiment(), offset_s=100.0)
        self.assertEqual(len(points), 10)
        self.assertEqual(points[0]["speaker"], 0)
        self.assertEqual(points[9]["speaker"], 1)
        self.assertAlmostEqual(points[0]["time"], 100.0)

    def test_summary_per_speaker_and_average(self):
        points = tr.extract_sentiment_points(self._raw_with_sentiment())
        summary = tr.summarize_sentiment(points)
        self.assertEqual(summary["per_speaker"][0]["sentiment"], "positive")
        self.assertEqual(summary["per_speaker"][1]["sentiment"], "negative")
        # 6 words at +0.8, 4 at -0.6 -> weighted mean +0.24
        self.assertAlmostEqual(summary["average"]["sentiment_score"], 0.24, places=5)
        self.assertEqual(len(summary["timeline"]), 12)

    def test_speaker_mapping_applied(self):
        points = tr.extract_sentiment_points(self._raw_with_sentiment(),
                                             speaker_mapping={0: 7, 1: 8})
        self.assertEqual({p["speaker"] for p in points}, {7, 8})

    def test_no_sentiment_block(self):
        self.assertEqual(tr.extract_sentiment_points(make_raw([utt(0, 0, 1, "hola")])), [])
        self.assertIsNone(tr.summarize_sentiment([]))

    def test_emoji_timeline(self):
        summary = tr.summarize_sentiment(tr.extract_sentiment_points(self._raw_with_sentiment()))
        line = tr.sentiment_timeline_emoji(summary)
        self.assertEqual(len(line), 12)
        self.assertIn("😊", line)


# ---------------------------------------------------------------------------
# Fallback / retry chain (mocked HTTP)
# ---------------------------------------------------------------------------

class TestDeepgramFallback(unittest.TestCase):
    def setUp(self):
        fd, self.audio_path = tempfile.mkstemp(suffix=".mp3")
        os.write(fd, b"fake-mp3-bytes")
        os.close(fd)
        patcher = mock.patch.object(tr.time, "sleep", lambda s: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(os.unlink, self.audio_path)

    def _good_raw(self):
        return make_raw([utt(0, 0, 5, "una transcripción suficientemente larga")],
                        duration=5)

    def test_model_fallback_on_http_error(self):
        calls = []

        def fake_post(url, params=None, headers=None, data=None, timeout=None):
            calls.append(params)
            if params.get("model") == "nova-3":
                return FakeResponse(400, text="bad model combo")
            return FakeResponse(200, self._good_raw())

        with mock.patch.object(tr.requests, "post", side_effect=fake_post):
            out = tr.transcribe_with_deepgram(self.audio_path, "key", language="es", diarize=True)
        self.assertIn("[Speaker 0]:", out)
        self.assertGreater(len(calls), 1)

    def test_first_attempt_uses_nova3_and_v2_diarizer(self):
        calls = []

        def fake_post(url, params=None, headers=None, data=None, timeout=None):
            calls.append(dict(params))
            return FakeResponse(200, self._good_raw())

        with mock.patch.object(tr.requests, "post", side_effect=fake_post):
            tr.transcribe_with_deepgram(self.audio_path, "key", language="es", diarize=True)
        first = calls[0]
        self.assertEqual(first.get("model"), "nova-3")
        self.assertEqual(first.get("language"), "es")
        self.assertEqual(first.get("diarize_model"), "latest")
        self.assertEqual(first.get("utterances"), "true")
        self.assertEqual(first.get("smart_format"), "true")

    def test_diarize_model_falls_back_to_legacy_param(self):
        calls = []

        def fake_post(url, params=None, headers=None, data=None, timeout=None):
            calls.append(dict(params))
            if "diarize_model" in params:
                return FakeResponse(400, text="unknown parameter diarize_model")
            return FakeResponse(200, self._good_raw())

        with mock.patch.object(tr.requests, "post", side_effect=fake_post):
            out = tr.transcribe_with_deepgram(self.audio_path, "key", language="es",
                                              diarize=True)
        self.assertIn("[Speaker 0]:", out)
        self.assertTrue(any(c.get("diarize") == "true" for c in calls))

    def test_retry_on_429(self):
        responses = [FakeResponse(429, text="slow down"), FakeResponse(200, self._good_raw())]
        with mock.patch.object(tr.requests, "post", side_effect=lambda *a, **k: responses.pop(0)):
            out = tr.transcribe_with_deepgram(self.audio_path, "key", language="es")
        self.assertIn("transcripción", out)
        self.assertEqual(responses, [])  # both consumed

    def test_features_stripped_on_400(self):
        calls = []

        def fake_post(url, params=None, headers=None, data=None, timeout=None):
            calls.append(dict(params))
            if "sentiment" in params:
                return FakeResponse(400, text="sentiment unsupported")
            return FakeResponse(200, self._good_raw())

        with mock.patch.object(tr.requests, "post", side_effect=fake_post):
            out = tr.transcribe_with_deepgram(
                self.audio_path, "key", language="en",
                features={"sentiment": "true"},
            )
        self.assertIn("transcripción", out)
        self.assertTrue(any("sentiment" not in c for c in calls))

    def test_english_only_features_not_sent_for_spanish(self):
        calls = []

        def fake_post(url, params=None, headers=None, data=None, timeout=None):
            calls.append(dict(params))
            return FakeResponse(200, self._good_raw())

        with mock.patch.object(tr.requests, "post", side_effect=fake_post):
            tr.transcribe_with_deepgram(
                self.audio_path, "key", language="es",
                features={"sentiment": "true", "utterances": "true"},
            )
        first_call = calls[0]
        self.assertNotIn("sentiment", first_call)
        self.assertEqual(first_call.get("utterances"), "true")

    def test_raw_mode_uses_fallback_on_empty_result(self):
        empty = make_raw(utterances=None, transcript="", duration=1)
        calls = []

        def fake_post(url, params=None, headers=None, data=None, timeout=None):
            calls.append(dict(params))
            if len(calls) == 1:
                return FakeResponse(200, empty)
            return FakeResponse(200, self._good_raw())

        with mock.patch.object(tr.requests, "post", side_effect=fake_post):
            raw = tr.transcribe_with_deepgram(self.audio_path, "key", language="es",
                                              return_raw=True)
        self.assertGreater(len(calls), 1)
        self.assertIn("transcripción", tr.get_transcript(raw))

    def test_all_attempts_fail(self):
        with mock.patch.object(tr.requests, "post",
                               side_effect=lambda *a, **k: FakeResponse(401, text="bad key")):
            with self.assertRaises(tr.DeepgramError):
                tr.transcribe_with_deepgram(self.audio_path, "key", language="es")
            raw = tr.transcribe_with_deepgram(self.audio_path, "key", language="es",
                                              return_raw=True)
        self.assertIn("error", raw)

    def test_missing_api_key(self):
        with self.assertRaises(EnvironmentError):
            tr.transcribe_with_deepgram(self.audio_path, "", language="es")


class TestReport(unittest.TestCase):
    def test_report_contains_sections(self):
        utterances = [utt(0, 0, 30, "hola " * 50), utt(1, 30, 60, "adiós " * 50)]
        insights = tr.compute_speech_insights(utterances, language="es")
        result = {
            "transcript": "[Speaker 0]: hola\n[Speaker 1]: adiós",
            "insights": insights,
            "sentiment": None,
            "warnings": ["aviso de prueba"],
            "model_used": "nova-2", "detected_language": "es", "duration": 60,
        }
        report = tr.build_report(result, filename="reunion.mp3")
        self.assertIn("INFORME DE TRANSCRIPCIÓN — reunion.mp3", report)
        self.assertIn("ANÁLISIS DE CONVERSACIÓN", report)
        self.assertIn("aviso de prueba", report)
        self.assertIn("[Speaker 0]: hola", report)


if __name__ == "__main__":
    unittest.main()
