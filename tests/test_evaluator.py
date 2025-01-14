from report_gen_eval import ModelProvider
from report_gen_eval.evaluator import empty_response, check_citations_relevance_detail, process_w_citations


def test_empty_response():
    assert (empty_response('sent') ==
            {"sentence": 'sent',
             "evaluation_path": [],
             "matched_nuggets": [],
             "score": 0,
             "citation_details": {
                 "has_citations": False,
                 "citations": [],
             },
             "evaluation_details": {
                 "is_negative": None,
                 "requires_citation": None,
                 "is_first_instance": None,
                 "model_responses": [],
             }})


def test_check_citations_relevance_detail_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            ['this is a test citation'],
                                            ModelProvider.YES) == ['RELEVANT']


def test_check_citations_relevance_detail_not_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            ['this is a test citation'],
                                            ModelProvider.NO) == ['NOT_RELEVANT']


def test_check_citations_relevance_detail_many_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            ['this is a test citation',
                                             'this is another test citation',
                                             'this is yet another test citation'],
                                            ModelProvider.YES) == ['RELEVANT', 'RELEVANT', 'RELEVANT']


def test_process_w_citations():
    assert process_w_citations(['one citation',
                                                 'another citation',
                                                 'yet another citation'],
                               '',
                               None,
                               ModelProvider.YES,
                               empty_response('this is a test sentence'),
                               'this is a test sentence'
                               ) == {"sentence": 'this is a test sentence',
                                     "evaluation_path": [],
                                     "matched_nuggets": [],
                                     "score": 3,
                                     "citation_details": {
                                         "has_citations": False,
                                         "citations": [
                                             {
                                                 "text": 'one citation',
                                                 "relevance": 'RELEVANT',
                                             },
                                             {
                                                 "text": 'another citation',
                                                 "relevance": 'RELEVANT',
                                             },
                                             {
                                                 "text": 'yet another citation',
                                                 "relevance": 'RELEVANT',
                                             }
                                         ],
                                     },
                                     "evaluation_details": {
                                         "is_negative": None,
                                         "requires_citation": None,
                                         "is_first_instance": None,
                                         "model_responses": [],
                                     }}
