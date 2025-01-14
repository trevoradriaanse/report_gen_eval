from report_gen_eval.utils import load_jsonl, get_model_response, ModelProvider


def test_get_model_response_yes():
    assert get_model_response('', '', ModelProvider.YES) == 'YES'


def test_get_model_response_no():
    assert get_model_response('', '', ModelProvider.NO) == 'NO'

