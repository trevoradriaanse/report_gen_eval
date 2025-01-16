from report_gen_eval import ModelProvider
from report_gen_eval.evaluator import empty_response, check_citations_relevance_detail, process_w_citations, \
    process_citation_relevancy, load_nuggets, filter_nuggets, check_nugget_matches, process_nuggets
from report_gen_eval.utils import load_jsonl


def test_empty_response():
    assert (empty_response('sent') ==
            {"segment_type": 'sentence',
             "text": 'sent',
             "citations": [],
             "judgments": [],
             })


def test_check_citations_relevance_detail_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            [{"doc_id": "D1", "text": 'this is a test citation'}],
                                            ModelProvider.YES) == ['RELEVANT']


def test_check_citations_relevance_detail_not_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            [{"doc_id": "D1", "text": 'this is a test citation'}],
                                            ModelProvider.NO) == ['NOT_RELEVANT']


def test_check_citations_relevance_detail_many_relevant():
    assert check_citations_relevance_detail('this is a test sentence',
                                            [{"doc_id": "D1", "text": 'this is a test citation'},
                                             {"doc_id": "D2", "text": 'this is another citation'},
                                             {"doc_id": "D3", "text": 'this is yet another test citation'}],
                                            ModelProvider.YES) == ['RELEVANT', 'RELEVANT', 'RELEVANT']


def test_process_citation_relevancy():
    assert process_citation_relevancy([{"doc_id": "D1", "text": 'one citation'},
                                       {"doc_id": "D2", "text": 'another citation'},
                                       {"doc_id": "D3", "text": 'yet another citation'}],
                                      '',
                                      ModelProvider.YES,
                                      empty_response('this is a test sentence'),
                                      'this is a test sentence'
                                      ) == {
               "segment_type": "sentence",
               "text": 'this is a test sentence',
               "citations": ["D1", "D2", "D3"],
               "judgments": [
                   {
                       "judgment_type_id": "Cited document is relevant?",
                       "response": {"doc_id": "D1", "answer": "RELEVANT", },
                       "evaluator": ModelProvider.YES,
                       "provenance": "D1",
                   },
                   {
                       "judgment_type_id": "Cited document is relevant?",
                       "response": {"doc_id": "D2", "answer": "RELEVANT", },
                       "evaluator": ModelProvider.YES,
                       "provenance": "D2",
                   },
                   {
                       "judgment_type_id": "Cited document is relevant?",
                       "response": {"doc_id": "D3", "answer": "RELEVANT", },
                       "evaluator": ModelProvider.YES,
                       "provenance": "D3",
                   },
               ],
           }


def test_load_nugget():
    assert load_nuggets('assets/avengers_nuggets_fix.jsonl',
                        load_jsonl('assets/avengers_report.jsonl')[0],
                        False) == [{'gold_answers': [{'answer': '2010', 'citations': ['D1']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_bcb4c843099c4f6688b8f4ff0be297e1',
                                   'question_text': 'When did Avatar first become the highest grossing film?'},
                                  {'gold_answers': [{'answer': '2019',
                                                     'citations': ['D1',
                                                                   'D2',
                                                                   'D3',
                                                                   'D8',
                                                                   'D10',
                                                                   'D12',
                                                                   'D13']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_7e37bab4e7bb4e02bf11a15cb636d24e',
                                   'question_text': 'When did Avengers: Endgame become the highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'July 2019', 'citations': ['D3', 'D12', 'D13']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_c8279ff6b8864a98ba1b0221d2d77e6e',
                                   'question_text': 'When did Avengers: Endgame become the highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'July 20, 2019', 'citations': ['D3']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_7f3c4472b8e04a1c9be6dcd6e38a27e7',
                                   'question_text': 'When did Avengers: Endgame become the highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'July 21, 2019', 'citations': ['D13']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_edb02abb402c4d0dac9c173068d722d5',
                                   'question_text': 'When did Avengers: Endgame become the highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'Added six minutes of additional footage',
                                                     'citations': ['D4']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_0ef6eb7b2f9b493d956ba04f79a4fbd3',
                                   'question_text': 'What did studio executives do to the Avengers: Endgame '
                                                    'film to become the highest grossing film?'},
                                  {'gold_answers': [{'answer': 'Added footage', 'citations': ['D4']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_966e94341498417998b6bbba934544e1',
                                   'question_text': 'What did studio executives do to the Avengers: Endgame '
                                                    'film to become the highest grossing film?'},
                                  {'gold_answers': [{'answer': 'Added 6 minutes', 'citations': ['D4']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_498ecdb4e3044bc280e0daed5d0f1c5f',
                                   'question_text': 'What did studio executives do to the Avengers: Endgame '
                                                    'film to become the highest grossing film?'},
                                  {'gold_answers': [{'answer': 'Additional footage at the end of the film',
                                                     'citations': ['D14']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_96feb31765c64bed83f2d951f03d81d3',
                                   'question_text': 'What did studio executives do to the Avengers: Endgame '
                                                    'film to become the highest grossing film?'},
                                  {'gold_answers': [{'answer': '2021',
                                                     'citations': ['D1', 'D2', 'D6', 'D7', 'D9', 'D11']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_56f4cc852e0f472a869530246f08e518',
                                   'question_text': 'When did Avatar retake the title of highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'March 2021',
                                                     'citations': ['D1', 'D6', 'D7', 'D9', 'D11']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_915efe95bf594b369bdc7c45e9bf32a6',
                                   'question_text': 'When did Avatar retake the title of highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'March 13, 2021',
                                                     'citations': ['D1', 'D6', 'D9']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_a5f2496169e64002bec48e96d23593bb',
                                   'question_text': 'When did Avatar retake the title of highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'Two years after the Avengers: Endgame became '
                                                               'the highest grossing film',
                                                     'citations': ['D2']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_bd58e1aa41b64078b3d22712fc6a9470',
                                   'question_text': 'When did Avatar retake the title of highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'Re-release in Mainland China',
                                                     'citations': ['D1', 'D2', 'D5', 'D7', 'D8', 'D9', 'D10']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_f0f903998f934b9da1b2a39d39fc41c9',
                                   'question_text': 'What event led to Avatar becoming the highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'Re-release in China',
                                                     'citations': ['D1',
                                                                   'D2',
                                                                   'D5',
                                                                   'D6',
                                                                   'D7',
                                                                   'D8',
                                                                   'D9',
                                                                   'D10']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_9babc589d9df4aa5b4c0b25c6d585724',
                                   'question_text': 'What event led to Avatar becoming the highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'Release in Mainland China for a second time',
                                                     'citations': ['D1',
                                                                   'D2',
                                                                   'D5',
                                                                   'D6',
                                                                   'D7',
                                                                   'D8',
                                                                   'D9',
                                                                   'D10']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_dd24d02244fa4dc38cc8dfd991aa5d28',
                                   'question_text': 'What event led to Avatar becoming the highest grossing '
                                                    'film?'},
                                  {'gold_answers': [{'answer': 'Returned to theaters in China',
                                                     'citations': ['D11']}],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_d89a5f78372d45bf926f4e42be9164c3',
                                   'question_text': 'What event led to Avatar becoming the highest grossing '
                                                    'film?'}]


def test_load_nugget_example():
    assert load_nuggets('assets/example_nuggets.jsonl',
                        load_jsonl('assets/example_input_one_only.jsonl')[0],
                        False) == [{'gold_answers': ['3.7%'],
                                   'info': {'importance': 'vital', 'used': False},
                                   'query_id': '300',
                                   'question_id': '300_test',
                                   'question_text': 'How much did suicides rise by in 2020?'}]


def test_filter_nuggets():
    assert filter_nuggets(load_nuggets('assets/avengers_nuggets_fix.jsonl',
                                       load_jsonl('assets/avengers_report.jsonl')[0],
                                       False),
                          "D2") == [{'gold_answers': [{'answer': '2019',
                                                       'citations': ['D1',
                                                                     'D2',
                                                                     'D3',
                                                                     'D8',
                                                                     'D10',
                                                                     'D12',
                                                                     'D13']}],
                                     'info': {'importance': 'vital', 'used': False},
                                     'query_id': '300',
                                     'question_id': '300_7e37bab4e7bb4e02bf11a15cb636d24e',
                                     'question_text': 'When did Avengers: Endgame become the highest grossing '
                                                      'film?'},
                                    {'gold_answers': [{'answer': '2021',
                                                       'citations': ['D1', 'D2', 'D6', 'D7', 'D9', 'D11']}],
                                     'info': {'importance': 'vital', 'used': False},
                                     'query_id': '300',
                                     'question_id': '300_56f4cc852e0f472a869530246f08e518',
                                     'question_text': 'When did Avatar retake the title of highest grossing '
                                                      'film?'},
                                    {'gold_answers': [{'answer': 'Two years after the Avengers: Endgame became '
                                                                 'the highest grossing film',
                                                       'citations': ['D2']}],
                                     'info': {'importance': 'vital', 'used': False},
                                     'query_id': '300',
                                     'question_id': '300_bd58e1aa41b64078b3d22712fc6a9470',
                                     'question_text': 'When did Avatar retake the title of highest grossing '
                                                      'film?'},
                                    {'gold_answers': [{'answer': 'Re-release in Mainland China',
                                                       'citations': ['D1', 'D2', 'D5', 'D7', 'D8', 'D9', 'D10']}],
                                     'info': {'importance': 'vital', 'used': False},
                                     'query_id': '300',
                                     'question_id': '300_f0f903998f934b9da1b2a39d39fc41c9',
                                     'question_text': 'What event led to Avatar becoming the highest grossing '
                                                      'film?'},
                                    {'gold_answers': [{'answer': 'Re-release in China',
                                                       'citations': ['D1',
                                                                     'D2',
                                                                     'D5',
                                                                     'D6',
                                                                     'D7',
                                                                     'D8',
                                                                     'D9',
                                                                     'D10']}],
                                     'info': {'importance': 'vital', 'used': False},
                                     'query_id': '300',
                                     'question_id': '300_9babc589d9df4aa5b4c0b25c6d585724',
                                     'question_text': 'What event led to Avatar becoming the highest grossing '
                                                      'film?'},
                                    {'gold_answers': [{'answer': 'Release in Mainland China for a second time',
                                                       'citations': ['D1',
                                                                     'D2',
                                                                     'D5',
                                                                     'D6',
                                                                     'D7',
                                                                     'D8',
                                                                     'D9',
                                                                     'D10']}],
                                     'info': {'importance': 'vital', 'used': False},
                                     'query_id': '300',
                                     'question_id': '300_dd24d02244fa4dc38cc8dfd991aa5d28',
                                     'question_text': 'What event led to Avatar becoming the highest grossing '
                                                      'film?'},
                                    ]


def test_filter_nuggets_D14():
    assert filter_nuggets(load_nuggets('assets/avengers_nuggets_fix.jsonl',
                                       load_jsonl('assets/avengers_report.jsonl')[0],
                                       False),
                            "D14") == [{'gold_answers': [{'answer': 'Additional footage at the end of the film',
                                                         'citations': ['D14']}],
                                         'info': {'importance': 'vital', 'used': False},
                                         'query_id': '300',
                                         'question_id': '300_96feb31765c64bed83f2d951f03d81d3',
                                         'question_text': 'What did studio executives do to the Avengers: Endgame '
                                                        'film to become the highest grossing film?'},
                                        ]


def test_check_nugget_matches():
    assert check_nugget_matches("this is a test sentence", load_nuggets('assets/example_nuggets_fix.jsonl',
                                                                        load_jsonl(
                                                                           'assets/example_input_one_only.jsonl')[0],
                                                                        False),
                                ModelProvider.YES) == [{
        "question_text": "How much did suicides rise by in 2020?",
        "matched_answer": "3.7%",
        "importance": "vital"
    }]


def test_process_w_citations():
    results = empty_response('this is a test sentence')
    results['citations'] = [{"doc_id": 'D14', "text": 'this is a test citation'}]
    assert process_w_citations([{"doc_id": 'D14', "text": 'this is a test citation'}],
                               '',
                               filter_nuggets(load_nuggets('assets/avengers_nuggets_fix.jsonl',
                                                           load_jsonl('assets/avengers_report.jsonl')[0],
                                                           False),
                                              "D14"),
                               ModelProvider.YES,
                               results,
                               'this is a test sentence') == {
        "segment_type": "sentence",
        "text": 'this is a test sentence',
        "citations": ["D14"],
        "judgments": [
            {
                "judgment_type_id": "Cited document is relevant?",
                "response": {"doc_id": "D14", "answer": "RELEVANT", },
                "evaluator": ModelProvider.YES,
                "provenance": "D14",
            },
            {
                "judgment_type_id": "Sentence answers question?",
                "response": {"matched_nuggets": [{
                    "question_text": "What did studio executives do to the Avengers: Endgame film to become the highest grossing film?",
                    "matched_answer": "Additional footage at the end of the film",
                    "importance": "vital",
                },
                ]},
                "evaluator": ModelProvider.YES,
                "provenance": "D14",
            },
        ],
    }





def test_process_nuggets():
    results = empty_response('this is a test sentence')
    results['citations'] = ['D14']
    assert process_nuggets('',
                           filter_nuggets(load_nuggets('assets/avengers_nuggets_fix.jsonl',
                                                       load_jsonl('assets/avengers_report.jsonl')[0],
                                        False),
                                        "D14"),
                           ModelProvider.YES,
                           results,
                           'this is a test sentence') == {
        "segment_type": "sentence",
        "text": 'this is a test sentence',
        "citations": ["D14"],
        "judgments": [
            {
                "judgment_type_id": "Sentence answers question?",
                "response": {"matched_nuggets": [{
                    "question_text": "What did studio executives do to the Avengers: Endgame film to become the highest grossing film?",
                    "matched_answer": "Additional footage at the end of the film",
                    "importance": "vital",
                },
                ]},
                "evaluator": ModelProvider.YES,
                "provenance": "D14",
            },
        ],
    }
