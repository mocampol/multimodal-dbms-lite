from transaction.demo import run_demo


def test_demo_shows_race_and_its_resolution():
    results = run_demo(echo=False)

    assert results["lost_update_autocommit"] in (130, 150), "without transactions one deposit is lost"
    assert results["lost_update_transactions"] == 180
    assert results["dirty_read"] == 100
    assert sorted(results["deadlock"].values()) == ["abort", "commit"]
    assert sum(results["deadlock_balances"]) == 200, "the victim's transfer must be fully rolled back"
