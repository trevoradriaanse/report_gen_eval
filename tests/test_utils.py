from report_gen_eval.utils import load_jsonl


def test_load_jsonl():
    assert len(load_jsonl("/home/hltcoe/mmason/dev/report_gen_eval/tests/assets/example_input.jsonl")) == 6
