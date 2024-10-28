
from dataclasses import dataclass, field
from typing import Dict, List
import time
from datetime import datetime


@dataclass
class APICall:
    timestamp: datetime
    model: str
    tokens_in: int
    tokens_out: int
    endpoint: str  # 'completion' or 'embedding'

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def estimated_cost(self) -> float:
        # Current OpenAI pricing (as of April 2024)
        costs = {
            'gpt-4-turbo-preview': {
                'input': 0.01,  # per 1K tokens
                'output': 0.03  # per 1K tokens
            },
            'gpt-3.5-turbo': {
                'input': 0.0005,
                'output': 0.0015
            },
            'text-embedding-3-small': {
                'input': 0.00002,
                'output': 0.00002
            }
        }

        if self.model not in costs:
            return 0.0

        model_costs = costs[self.model]
        input_cost = (self.tokens_in / 1000) * model_costs['input']
        output_cost = (self.tokens_out / 1000) * model_costs['output']
        return input_cost + output_cost


@dataclass
class ProcessingSession:
    start_time: datetime = field(default_factory=datetime.now)
    api_calls: List[APICall] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(call.estimated_cost for call in self.api_calls)

    @property
    def total_tokens(self) -> Dict[str, int]:
        totals = {}
        for call in self.api_calls:
            if call.model not in totals:
                totals[call.model] = {'input': 0, 'output': 0, 'total': 0}
            totals[call.model]['input'] += call.tokens_in
            totals[call.model]['output'] += call.tokens_out
            totals[call.model]['total'] += call.total_tokens
        return totals

    def generate_report(self) -> str:
        duration = datetime.now() - self.start_time
        token_usage = self.total_tokens

        report = [
            "DDQPro Processing Cost Report",
            "=" * 40,
            f"Duration: {duration}",
            f"Total Cost: ${self.total_cost:.2f}",
            "\nToken Usage by Model:",
        ]

        for model, tokens in token_usage.items():
            report.extend([
                f"\n{model}:",
                f"  Input tokens:  {tokens['input']:,}",
                f"  Output tokens: {tokens['output']:,}",
                f"  Total tokens:  {tokens['total']:,}"
            ])

        return "\n".join(report)


class CostTracker:
    def __init__(self):
        self.current_session = ProcessingSession()

    def track_call(self, model: str, tokens_in: int, tokens_out: int, endpoint: str):
        """Track an API call"""
        call = APICall(
            timestamp=datetime.now(),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            endpoint=endpoint
        )
        self.current_session.api_calls.append(call)

    def get_current_costs(self) -> float:
        """Get costs for current session"""
        return self.current_session.total_cost

    def get_report(self) -> str:
        """Get detailed cost report"""
        return self.current_session.generate_report()


