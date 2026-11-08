"""Comprehensive CPU-compatible verification suite for Qwen2.5-3B QLoRA + Local RAG.

Runs with Python's standard library ``unittest`` without requiring ``torch`` or ``faiss``:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_utils import (  # noqa: E402
    build_rag_prompt,
    context_block,
    filter_and_rank_hits,
    init_kb,
    is_conversational_query,
    retrieve,
)


class TestKnowledgeBaseIntegrity(unittest.TestCase):
    """Verify kb/facts.jsonl, kb/meta.json, and kb/index.faiss schema and alignment."""

    def setUp(self) -> None:
        self.kb_dir = REPO_ROOT / "kb"
        self.facts_path = self.kb_dir / "facts.jsonl"
        self.meta_path = self.kb_dir / "meta.json"
        self.index_path = self.kb_dir / "index.faiss"

    def test_kb_files_exist_and_non_empty(self) -> None:
        for path in (self.facts_path, self.meta_path, self.index_path):
            self.assertTrue(path.exists(), f"Missing KB artifact: {path}")
            self.assertGreater(path.stat().st_size, 100, f"Empty KB artifact: {path}")

    def test_facts_and_meta_exact_alignment(self) -> None:
        facts = []
        with open(self.facts_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    self.assertIn("id", row)
                    self.assertIn("title", row)
                    self.assertIn("text", row)
                    facts.append(row)

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(len(facts), 3)
        self.assertEqual(meta["ids"], [r["id"] for r in facts])
        self.assertEqual(meta["titles"], [r["title"] for r in facts])
        self.assertEqual(meta["texts"], [r["text"] for r in facts])

    def test_canberra_1913_fact_present(self) -> None:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("au_canberra_1913", meta["ids"])
        idx = meta["ids"].index("au_canberra_1913")
        self.assertEqual(meta["titles"][idx], "Canberra founding")
        self.assertIn("1913", meta["texts"][idx])
        self.assertIn("Canberra", meta["texts"][idx])


class TestEmpiricalEvidenceAndMetrics(unittest.TestCase):
    """Verify evidence/*.json artifacts and metric consistency."""

    def setUp(self) -> None:
        self.evidence_dir = REPO_ROOT / "evidence"

    def test_eval_summary_metrics_and_canberra_case(self) -> None:
        path = self.evidence_dir / "EVAL_SUMMARY.json"
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["num_rows"], 30)
        self.assertAlmostEqual(data["avg_len_base_words"], 68.2, places=1)
        self.assertAlmostEqual(data["avg_len_adapter_words"], 39.0, places=1)
        self.assertLess(data["avg_len_adapter_words"], data["avg_len_base_words"])
        self.assertAlmostEqual(data["word_reduction_pct"], -42.8, places=1)

        self.assertAlmostEqual(data["rougeL_base"], 0.2655, places=4)
        self.assertAlmostEqual(data["rougeL_adapter"], 0.3351, places=4)
        self.assertGreater(data["rougeL_adapter"], data["rougeL_base"])
        self.assertAlmostEqual(data["rougeL_relative_gain_pct"], 26.2, places=1)

        self.assertAlmostEqual(data["peak_vram_gb"], 3.88, places=2)
        self.assertLess(data["peak_vram_gb"], 4.0)
        self.assertAlmostEqual(data["avg_latency_sec"], 3.43, places=2)

        canberra = data["example_canberra"]
        self.assertEqual(canberra["idx"], 187)
        self.assertIn("1913", canberra["gold"])
        self.assertIn("1908", canberra["base"])
        self.assertIn("1908", canberra["adapter"])

    def test_trainer_state_steps150_trajectory(self) -> None:
        path = self.evidence_dir / "trainer_state_steps150.json"
        self.assertTrue(path.exists())
        ts = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(ts["max_steps"], 150)
        self.assertEqual(ts["global_step"], 150)
        self.assertEqual(ts["logging_steps"], 20)

        history = ts["log_history"]
        self.assertEqual([h["step"] for h in history], [20, 40, 60, 80, 100, 120, 140])
        self.assertAlmostEqual(history[0]["loss"], 2.1785, places=4)
        self.assertAlmostEqual(history[-1]["loss"], 1.6605, places=4)
        self.assertLess(history[-1]["loss"], history[0]["loss"])

        self.assertAlmostEqual(history[0]["mean_token_accuracy"], 0.5703824, places=5)
        self.assertAlmostEqual(history[-1]["mean_token_accuracy"], 0.6679676, places=5)
        self.assertGreater(history[-1]["mean_token_accuracy"], history[0]["mean_token_accuracy"])

    def test_adapter_config_hyperparameters(self) -> None:
        path = self.evidence_dir / "adapter_config.json"
        self.assertTrue(path.exists())
        cfg = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(cfg["base_model_name_or_path"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(cfg["peft_type"], "LORA")
        self.assertEqual(cfg["task_type"], "CAUSAL_LM")
        self.assertEqual(cfg["r"], 8)
        self.assertEqual(cfg["lora_alpha"], 16)
        self.assertAlmostEqual(cfg["lora_dropout"], 0.05, places=3)
        self.assertEqual(
            set(cfg["target_modules"]),
            {"q_proj", "k_proj", "v_proj", "o_proj"},
        )

    def test_ablation_4way_report_and_canberra_case_study(self) -> None:
        ablation_path = self.evidence_dir / "ablation_4way_report.json"
        case_path = self.evidence_dir / "canberra_case_study.json"
        self.assertTrue(ablation_path.exists())
        self.assertTrue(case_path.exists())

        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        self.assertEqual(ablation["steps"], [20, 40, 60, 80, 100, 120, 140, 150])
        self.assertEqual(len(ablation["configurations"]), 4)
        cfg_ids = [c["id"] for c in ablation["configurations"]]
        self.assertEqual(
            cfg_ids,
            [
                "base_zero_shot",
                "base_naive_rag",
                "qlora_adapter_no_rag",
                "qlora_confidence_gated_rag",
            ],
        )

        case = json.loads(case_path.read_text(encoding="utf-8"))
        self.assertEqual(case["gold_reference"]["verified_founding_year"], 1913)
        modes = case["modes"]
        self.assertFalse(modes["mode_1_base_model"]["factually_correct"])
        self.assertFalse(modes["mode_2_qlora_adapter_no_rag"]["factually_correct"])
        self.assertFalse(modes["mode_3_stage2_continual_patch"]["factually_correct"])
        self.assertEqual(
            modes["mode_3_stage2_continual_patch"]["patch_dataset_composition"]["replay_density_ratio"],
            "1/1016",
        )
        self.assertTrue(modes["mode_4_qlora_confidence_gated_rag"]["factually_correct"])
        self.assertEqual(modes["mode_4_qlora_confidence_gated_rag"]["reported_year"], 1913)


class TestSvgPlotsAndChecksums(unittest.TestCase):
    """Verify SVG plots are well-formed XML and CHECKSUMS.txt matches disk."""

    def test_svg_plots_valid_xml(self) -> None:
        plots_dir = REPO_ROOT / "evidence" / "plots"
        expected_plots = [
            "training_convergence.svg",
            "ablation_4way_comparison.svg",
            "vram_budget_4gb.svg",
        ]
        for name in expected_plots:
            svg_path = plots_dir / name
            self.assertTrue(svg_path.exists(), f"Missing SVG plot: {svg_path}")
            tree = ET.parse(svg_path)
            root = tree.getroot()
            self.assertTrue(root.tag.endswith("svg"), f"Invalid root tag in {name}: {root.tag}")

    def test_checksums_manifest_matches_files(self) -> None:
        checksums_path = REPO_ROOT / "CHECKSUMS.txt"
        self.assertTrue(checksums_path.exists())
        verified_count = 0
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            self.assertGreaterEqual(len(parts), 4, f"Malformed checksum line: {line}")
            sha256_expected, md5_expected, size_expected, rel_path = (
                parts[0],
                parts[1],
                int(parts[2]),
                parts[3],
            )
            if rel_path.startswith("zip://"):
                continue
            target = REPO_ROOT / rel_path
            self.assertTrue(target.exists(), f"Missing checksummed file: {rel_path}")
            raw = target.read_bytes()
            self.assertEqual(len(raw), size_expected, f"Byte size mismatch for {rel_path}")
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(),
                sha256_expected,
                f"SHA-256 mismatch for {rel_path}",
            )
            self.assertEqual(
                hashlib.md5(raw).hexdigest(),
                md5_expected,
                f"MD5 mismatch for {rel_path}",
            )
            verified_count += 1
        self.assertGreaterEqual(verified_count, 9)


class TestConversationalIntentRouter(unittest.TestCase):
    """Verify greeting detection fixes the 'hi'/'hey' substring bug."""

    def test_true_greetings_detected(self) -> None:
        greetings = [
            "hi",
            "Hi!",
            "hello!",
            "  hello  ",
            "hey",
            "thanks",
            "thank you!",
            "good morning",
            "how are you?",
            "bye",
            "goodbye.",
        ]
        for q in greetings:
            with self.subTest(query=q):
                self.assertTrue(
                    is_conversational_query(q),
                    f"Expected greeting {q!r} to be classified as conversational",
                )

    def test_factual_queries_with_hi_or_hey_substrings_not_misclassified(self) -> None:
        factual_queries = [
            "Which city is the capital of Australia?",
            "Explain this study technique",
            "Explain machine learning precision vs recall",
            "What are they studying?",
            "Tell me about Australian history and Canberra",
            "How does hierarchical clustering work?",
            "What is the capital of Australia and when was it founded?",
        ]
        for q in factual_queries:
            with self.subTest(query=q):
                self.assertFalse(
                    is_conversational_query(q),
                    f"Factual query {q!r} triggered false-positive conversational match!",
                )


class TestConfidenceGatedRagFiltering(unittest.TestCase):
    """Verify FAISS sentinel removal, score threshold gating, and rank re-indexing."""

    def setUp(self) -> None:
        with open(REPO_ROOT / "kb" / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def test_sentinel_and_out_of_bounds_removal(self) -> None:
        scores = [[0.82, 0.45, -1.0, 0.99]]
        indices = [[0, -1, 2, 99]]
        hits = filter_and_rank_hits(scores, indices, self.meta, score_threshold=0.30)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["rank"], 1)
        self.assertEqual(hits[0]["id"], "au_canberra_1913")
        self.assertAlmostEqual(hits[0]["score"], 0.82, places=4)

    def test_score_threshold_gating_and_contiguous_reranking(self) -> None:
        scores = [0.78, 0.21, 0.44]
        indices = [0, 1, 2]
        hits = filter_and_rank_hits(scores, indices, self.meta, score_threshold=0.30)
        self.assertEqual(len(hits), 2)
        self.assertEqual([h["rank"] for h in hits], [1, 2])
        self.assertEqual([h["id"] for h in hits], ["au_canberra_1913", "study_methods"])

    def test_retrieve_and_build_rag_prompt_end_to_end(self) -> None:
        init_kb(REPO_ROOT / "kb")

        # 1. Grounded factual query on Canberra founding
        plan_canberra = build_rag_prompt(
            system_prompt="You are a concise assistant.",
            user_query="What is the capital of Australia and when was it founded?",
            use_rag=True,
            top_k=3,
            score_threshold=0.30,
        )
        self.assertEqual(plan_canberra["mode"], "rag_grounded")
        self.assertGreaterEqual(len(plan_canberra["hits"]), 1)
        self.assertEqual(plan_canberra["hits"][0]["id"], "au_canberra_1913")
        self.assertIn("1913", plan_canberra["user"])
        self.assertIn("CONTEXT:", plan_canberra["user"])

        # 2. Conversational greeting bypasses retrieval
        plan_hi = build_rag_prompt(
            system_prompt="You are a concise assistant.",
            user_query="hello!",
            use_rag=True,
            top_k=3,
            score_threshold=0.30,
        )
        self.assertEqual(plan_hi["mode"], "conversational")
        self.assertEqual(plan_hi["hits"], [])

        # 3. Out-of-domain query with strict_kb=True abstains cleanly
        plan_ood_strict = build_rag_prompt(
            system_prompt="You are a concise assistant.",
            user_query="What is the boiling point of nitrogen on Pluto?",
            use_rag=True,
            top_k=3,
            score_threshold=0.30,
            strict_kb=True,
        )
        self.assertEqual(plan_ood_strict["mode"], "rag_abstain")
        self.assertEqual(plan_ood_strict["hits"], [])

        # 4. Context block formatting
        formatted = context_block(plan_canberra["hits"], include_scores=True)
        self.assertIn("[1]", formatted)
        self.assertIn("au_canberra_1913", formatted)


class TestFastApiAndEvaluationPipeline(unittest.TestCase):
    """Verify FastAPI microservice handlers and evaluation summary aggregation."""

    def test_fastapi_health_model_info_retrieve_and_generate(self) -> None:
        from src.api import (
            get_health,
            get_model_info,
            handle_generate,
            handle_retrieve,
            set_custom_generator,
        )

        health = get_health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["kb"]["num_passages"], 3)

        info = get_model_info()
        self.assertEqual(info["base_model"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(info["adapter_config"]["r"], 8)
        self.assertEqual(info["adapter_config"]["lora_alpha"], 16)
        self.assertIn("q_proj", info["adapter_config"]["target_modules"])

        ret = handle_retrieve(
            query="What is the capital of Australia and when was it founded?",
            top_k=3,
            score_threshold=0.30,
        )
        self.assertEqual(ret["mode"], "rag_grounded")
        self.assertEqual(ret["hits"][0]["id"], "au_canberra_1913")

        try:
            set_custom_generator(
                lambda sys_p, usr_p, mnt, temp, dis: "The capital of Australia is Canberra, founded in 1913 [1]."
            )
            gen = handle_generate(
                query="What is the capital of Australia and when was it founded?",
                use_rag=True,
                score_threshold=0.30,
            )
            self.assertEqual(gen["mode"], "rag_grounded")
            self.assertIn("1913", gen["answer"])
        finally:
            set_custom_generator(None)

    def test_summarize_eval_rows_reconciliation(self) -> None:
        from src.summarize_eval import summarize_eval_rows

        rows = [
            {
                "idx": 187,
                "user": "What is the capital of Australia and when was it founded?",
                "gold": "The capital of Australia is Canberra. It was formally founded and named in 1913.",
                "base": "The capital of Australia is Canberra. It was founded in 1908 after long debate between Sydney and Melbourne.",
                "adapter": "The capital of Australia is Canberra. It was founded in 1908.",
                "adapter_rag": "The capital of Australia is Canberra, officially founded in 1913 [1].",
            }
        ]
        summary = summarize_eval_rows(rows)
        self.assertEqual(summary["num_rows"], 1)
        self.assertTrue(summary["has_canberra_case"])
        self.assertLess(summary["avg_words_adapter"], summary["avg_words_base"])
        self.assertIn("rougeL_adapter", summary)
        self.assertIn("rougeL_adapter_rag", summary)


if __name__ == "__main__":
    unittest.main()
