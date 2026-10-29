from review_intel.data.prepare import build_reviews, make_splits, normalize


def test_normalize():
    assert normalize("  a\r\nb\t\tc  ") == "a b c"


def test_build_reviews(raw_dir):
    df = build_reviews(raw_dir)
    assert df["review_id"].is_unique
    assert df["text"].str.len().min() >= 5  # "Ok" sai
    assert df["text"].iloc[0].startswith("Título. ")
    assert not df["text"].str.contains("\n").any()
    assert set(df["delivered_late"].dropna()) == {True, False}
    assert (df["customer_state"] == "SP").all()


def test_splits_are_disjoint_and_cover_everything(raw_dir):
    df = build_reviews(raw_dir)
    splits = make_splits(df, n_silver=20, n_gold=10)
    ids = [set(s["review_id"]) for s in splits.values()]
    assert sum(map(len, ids)) == len(df)
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(splits["gold_candidates"]) == 10 and len(splits["silver_pool"]) == 20
    # determinístico
    again = make_splits(df, n_silver=20, n_gold=10)
    assert list(again["gold_candidates"]["review_id"]) == list(splits["gold_candidates"]["review_id"])
