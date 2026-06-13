from kvcache.canonicalize import canonicalize, canonicalize_ablation


def test_boxed_and_option_canonicalization():
    assert canonicalize(r"work \boxed{1/2}") == "1/2"
    assert canonicalize("The answer is (C)") == "C"


def test_second_canonicalizer_is_distinct():
    assert canonicalize_ablation("  Answer   Is C ") == "answer is c"
