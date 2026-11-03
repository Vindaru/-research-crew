"""Its presence at the repo root makes pytest (any version) put this folder on the import path.

It also prints what the test run cost (live tests only; offline tests are free)."""


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls Claude and Tavily, costs money; needs RUN_LIVE_TESTS=1")


def pytest_terminal_summary(terminalreporter):
    from research_crew.cost import cost_summary, format_cost

    summary = cost_summary()
    if summary["usd"]:
        terminalreporter.write_line(format_cost(summary))
