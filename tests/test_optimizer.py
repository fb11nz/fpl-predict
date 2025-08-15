from src.fpl_predict.transfer.optimizer import optimize_transfers
def test_optimizer_runs():
    plan = optimize_transfers()
    assert "picked_ids" in plan and len(plan["picked_ids"]) == 15
