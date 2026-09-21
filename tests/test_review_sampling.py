from detected_pipeline.review.sampling import plan_review, run_sampled_review

CFG = {"high_fraction": .2, "middle_fraction": .4, "middle_rate": .1, "middle_min": 5, "low_rate": .02, "low_min": 2}


def test_plan_segments_by_score_and_samples_with_minimums():
    scores = [i / 100 for i in range(100)]
    plan = plan_review(scores, CFG, "seed")
    assert len(plan["segments"]["high"]) == 20 and len(plan["segments"]["middle"]) == 40 and len(plan["segments"]["low"]) == 40
    assert set(plan["segments"]["high"]) == set(range(80, 100))       # highest scores
    assert plan["sampled"]["high"] == plan["segments"]["high"]         # full review
    assert len(plan["sampled"]["middle"]) == 5 and len(plan["sampled"]["low"]) == 2   # minimums apply (10 % of 40 = 4 -> 5)
    assert plan_review(scores, CFG, "seed") == plan                    # deterministic


def test_segment_escalates_to_full_review_when_sample_finds_ng():
    scores = [i / 100 for i in range(100)]
    hidden_ng = {5, 50}   # one in the low segment, one in the middle segment
    plan = plan_review(scores, CFG, "seed")
    calls = []
    result = run_sampled_review(scores, CFG, "seed", lambda i: (calls.append(i) or i in hidden_ng))
    low_sampled = set(plan["sampled"]["low"]); middle_sampled = set(plan["sampled"]["middle"])
    assert result["stats"]["low"]["escalated"] == (5 in low_sampled)
    assert result["stats"]["middle"]["escalated"] == (50 in middle_sampled)
    for name in ("low", "middle"):
        if result["stats"][name]["escalated"]:
            assert result["stats"][name]["reviewed"] == result["stats"][name]["members"]
    assert len(result["pseudo_ok"]) + len(result["reviewed"]) == 100
    assert not set(result["pseudo_ok"]) & set(plan["segments"]["high"])


def test_all_pseudo_ok_when_samples_are_clean():
    scores = [i / 100 for i in range(50)]
    result = run_sampled_review(scores, CFG, "s", lambda i: False)
    assert len(result["reviewed"]) == 10 + 5 + 2 and len(result["pseudo_ok"]) == 33
