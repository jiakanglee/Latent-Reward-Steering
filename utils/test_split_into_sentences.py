"""
Tests for the split_into_sentences function.

Run this file directly to execute all tests:
    python test_split_into_sentences.py
"""

from utils import split_into_sentences
import re

def test_basic_sentence_splitting():
    """Test basic sentence splitting with periods."""
    test_text = "This is the first sentence. This is the second sentence. This is the third sentence."
    result = split_into_sentences(test_text)
    expected = ["This is the first sentence.", "This is the second sentence.", "This is the third sentence."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Basic sentence splitting test passed")


def test_different_punctuation_marks():
    """Test splitting with different punctuation marks."""
    test_text = "What is this? This is amazing! I think so; maybe not."
    result = split_into_sentences(test_text, min_words=2)
    expected = ["What is this?", "This is amazing!", "I think so;", "maybe not."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Different punctuation marks test passed")


def test_short_sentence_filtering():
    """Test filtering out sentences with less than 3 words."""
    test_text = "This is a good sentence. Yes. No way. This is another good sentence."
    result = split_into_sentences(test_text)
    expected = ["This is a good sentence.", "This is another good sentence."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Short sentence filtering test passed")


def test_whitespace_handling():
    """Test proper trimming of leading/trailing spaces."""
    test_text = "  First sentence here.   Second sentence there.  "
    result = split_into_sentences(test_text)
    expected = ["First sentence here.", "Second sentence there."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Whitespace handling test passed")


def test_empty_string():
    """Test edge case for empty input."""
    test_text = ""
    result = split_into_sentences(test_text)
    expected = []
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Empty string test passed")


def test_single_sentence():
    """Test when no splitting is needed."""
    test_text = "This is a single sentence with more than three words"
    result = split_into_sentences(test_text)
    expected = ["This is a single sentence with more than three words"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Single sentence test passed")


def test_consecutive_punctuation():
    """Test handling of multiple punctuation marks."""
    test_text = "What?! This is crazy!! Are you sure? Yes I am."
    result = split_into_sentences(test_text, min_words=1)
    expected = ["What?", "!!", "This is crazy!!", "Are you sure?", "Yes I am."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Consecutive punctuation test passed")


def test_mixed_content():
    """Test combination of short and long sentences."""
    test_text = "I think this is good. Ok. But what about this longer sentence? Sure thing! No."
    result = split_into_sentences(test_text, min_words=2)
    expected = ["I think this is good.", "But what about this longer sentence?", "Sure thing!"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Mixed content test passed")


def test_only_whitespace_sentences():
    """Test sentences that become empty after stripping."""
    test_text = "Good sentence here.   .   Another good sentence."
    result = split_into_sentences(test_text)
    expected = ["Good sentence here.", "Another good sentence."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Whitespace-only sentences test passed")


def test_no_punctuation():
    """Test text without sentence-ending punctuation."""
    test_text = "This is a sentence without punctuation"
    result = split_into_sentences(test_text)
    expected = ["This is a sentence without punctuation"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ No punctuation test passed")


def test_decimal_numbers():
    """Test that decimal numbers don't cause sentence splits."""
    test_text = "Since the P-value is 0.075, which is less than 0.05, that suggests that the t-statistic is pretty large in magnitude, right?"
    result = split_into_sentences(test_text)
    expected = ["Since the P-value is 0.075, which is less than 0.05, that suggests that the t-statistic is pretty large in magnitude, right?"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Decimal numbers test passed")


def test_mixed_decimals_and_sentences():
    """Test sentences with decimal numbers and actual sentence breaks."""
    test_text = "The value is 3.14159 which is pi. The other value is 2.718 for Euler's number! What about 0.5 as a fraction?"
    result = split_into_sentences(test_text)
    expected = ["The value is 3.14159 which is pi.", "The other value is 2.718 for Euler's number!", "What about 0.5 as a fraction?"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Mixed decimals and sentences test passed")


def test_quoted_sentences_with_numbers():
    """Test splitting quoted sentences that end with numbers followed by periods."""
    test_text = 'Wait, the problem says "sales of $200,000 and returns of sales made in prior months of $5,000." So, the $200,000 is the current month\'s sales, and the $5,000 is the returns from prior months'
    result = split_into_sentences(test_text)
    expected = ['Wait, the problem says "sales of $200,000 and returns of sales made in prior months of $5,000."', "So, the $200,000 is the current month's sales, and the $5,000 is the returns from prior months"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Quoted sentences with numbers test passed")


def test_newline_splitting():
    """Test splitting sentences on newlines."""
    test_text = """To find 60% of 30, I first convert the percentage to a decimal by dividing 60 by 100, which gives 0.6.

Next, I multiply this decimal by 30 to determine the desired value"""
    result = split_into_sentences(test_text)
    expected = ["To find 60% of 30, I first convert the percentage to a decimal by dividing 60 by 100, which gives 0.6.", "Next, I multiply this decimal by 30 to determine the desired value"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Newline splitting test passed")


def test_newline_without_punctuation():
    """Test splitting on newlines when there's no sentence-ending punctuation."""
    test_text = """This is the first line with no punctuation

This is the second line also with no punctuation"""
    result = split_into_sentences(test_text)
    expected = ["This is the first line with no punctuation", "This is the second line also with no punctuation"]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Newline without punctuation test passed")


def test_quote_post_processing():
    """Test that quotes are properly moved to end of previous sentence after period splits."""
    test_text = 'He said "Hello world." She replied quickly.'
    result = split_into_sentences(test_text)
    expected = ['He said "Hello world."', 'She replied quickly.']
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Quote post-processing test passed")


def test_single_letter_abbreviations():
    """Test that single letter abbreviations don't cause sentence splits."""
    test_text = "First, the problem says there are 25,000 ribosomes in an E. coli cell."
    result = split_into_sentences(test_text)
    expected = ["First, the problem says there are 25,000 ribosomes in an E. coli cell."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Single letter abbreviations test passed")


def test_single_letter_with_exclamation():
    """Test that single letter mathematical expressions don't cause sentence splits."""
    test_text = "But in this problem, the subsets are ordered, so we need to multiply by k! to account for the different orderings."
    result = split_into_sentences(test_text)
    expected = ["But in this problem, the subsets are ordered, so we need to multiply by k! to account for the different orderings."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Single letter with exclamation test passed")


def test_normal_sentences_still_split():
    """Test that normal sentences still split correctly after single letter fix."""
    test_text = "This is a normal sentence. This should be split correctly! And so should this? Yes indeed it should."
    result = split_into_sentences(test_text)
    expected = ["This is a normal sentence.", "This should be split correctly!", "And so should this?", "Yes indeed it should."]
    assert result == expected, f"Test failed: expected {expected}, got {result}"
    print("✓ Normal sentences still split test passed")


def test_keeps_delimiters():
    """Test that delimiters are kept in the sentences."""
    test_text = """Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\nWhat is the percentage of children aged 13-15 in China who reported being in a physical fight at school, one or more times during the past 12 months as of 2015?\n\nStep by step answer:\nOkay, so I need to figure out the percentage of children aged 13-15 in China who reported being in a physical fight at school one or more times during the past 12 months as of 2015. Hmm, I'm not exactly sure where to start, but I'll try to break it down step by step.\n\nFirst, I think I should understand the context. China has a large population, and school-age children are a significant part of that. The age group in question is 13-15 years old, which is a time when kids are typically in middle school or junior high. Physical fights in schools can happen for various reasons, like bullying, disagreements, or other conflicts.\n\nI remember that the user mentioned a study from 2015, so I should look for data from that year. Maybe there's a report or survey conducted around that time. I'm not sure if it's from the Chinese Ministry of Education or another organization, but I think they might have such statistics.\n\nI should consider the methodology used in the study. Was it a national survey, or was it conducted in a specific region? The methodology affects the representativeness of the data. If it's a national survey, the results are more likely to be accurate for the entire population. If it's regional, the results might not apply to all parts of China.\n\nNext, I need to find the exact percentage. I'm not recalling the exact figure, but I think it's somewhere between 10% to 20%. I might be wrong, though. I should check if there are any reputable sources that have this information. Maybe organizations like the United Nations Children's Fund (UNICEF) or the Chinese Education Ministry publish such reports.\n\nWait, I should also think about how the data was collected. Was it through self-reported surveys, interviews with teachers, or some other method? Self-reported data might have biases, as students might underreport or overreport incidents. On the other hand, teacher reports might be more accurate but could be influenced by the teachers' perceptions or reporting biases.\n\nAnother thing to consider is the definition of a physical fight. Does it include minor shoving matches or only more violent incidents? The definition can significantly affect the statistics. If the study includes all physical altercations, the percentage might be higher, but if it's only severe fights, it might be lower.\n\nI also wonder about the sample size. A larger sample size would provide more accurate results, but if the sample is small, the percentage might not be representative of the entire population. I should look for studies that mention the sample size and how it was determined.\n\nAdditionally, cultural factors in China might influence the prevalence of fights in schools. For example, some schools might have stricter disciplinary policies, which could affect how often fights occur. Social economic factors could also play a role, as schools in different regions might have different levels of resources and support, leading to varying incidences of fights.\n\nI should also think about how the data is presented. Is it broken down by gender? I know that sometimes boys are more likely to be involved in physical fights than girls, so that could affect the percentage. If the study doesn't specify, it might be harder to interpret the results accurately.\n\nLastly, I need to ensure that the data is up-to-date. Since the question is about as of 2015, I should make sure that the study I'm referring to is from that year or the most recent one available if 2015 data isn't available.\n\nPutting it all together, I think the percentage is around 15%, but I'm not entirely sure. I should look for a reliable source that provides this specific statistic for the 13-15 age group in China as of 2015. If I can't find the exact figure, I might have to make an educated guess based on similar studies or trends in school violence over the years."""
    
    # Split using previous regex (with newlines)
    expected_sentences = re.split(r'(?<=[.!?;\n])', test_text)
    expected_sentences = [s.strip() for s in expected_sentences if s.strip()]
    expected_sentences = [s for s in expected_sentences if s]

    result = split_into_sentences(test_text, min_words=0)
    result = [s.strip() for s in result if s.strip()]
    result = [s for s in result if s]

    assert len(result) == len(expected_sentences), f"Test failed: expected {len(expected_sentences)} sentences, got {len(result)}"

    for i in range(len(result)):
        assert result[i] == expected_sentences[i], f"Test failed: expected {expected_sentences[i]}, got {result[i]}"
    
    print("✓ Keeps delimiters test passed")


def run_all_tests():
    """Run all test functions."""
    print("Running tests for split_into_sentences function...\n")
    
    test_functions = [
        test_basic_sentence_splitting,
        test_different_punctuation_marks,
        test_short_sentence_filtering,
        test_whitespace_handling,
        test_empty_string,
        test_single_sentence,
        test_consecutive_punctuation,
        test_mixed_content,
        test_only_whitespace_sentences,
        test_no_punctuation,
        test_decimal_numbers,
        test_mixed_decimals_and_sentences,
        test_quoted_sentences_with_numbers,
        test_newline_splitting,
        test_newline_without_punctuation,
        test_quote_post_processing,
        test_single_letter_abbreviations,
        test_single_letter_with_exclamation,
        test_normal_sentences_still_split,
        test_keeps_delimiters,
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    run_all_tests() 