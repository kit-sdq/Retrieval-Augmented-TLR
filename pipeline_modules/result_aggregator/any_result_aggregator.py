from .result_aggregator import ResultAggregator, TraceLink, ModuleConfiguration, ClassificationResult


class AnyResultAggregator(ResultAggregator):
    """Reports a trace link if any connection between source and target at the wanted granularities was classified.
    Granularities can only be
    This aggregator treats each source-target combination as a separate result."""
    __source_granularity: int
    __target_granularity: int

    def __init__(self, configuration: ModuleConfiguration):
        self.__source_granularity = int(configuration.args.setdefault('source_granularity', 0))
        self.__target_granularity = int(configuration.args.setdefault('target_granularity', 0))

    def aggregate(self, results: list[ClassificationResult]) -> set[TraceLink]:
        links: set[TraceLink] = set()
        for result in results:
            source = result.source
            while source.granularity > self.__source_granularity:
                source = source.parent

            for candidate in result.target:
                target = candidate
                while target.granularity > self.__target_granularity:
                    target = target.parent

                links.add(TraceLink(source=source.identifier, target=target.identifier))

        return links
