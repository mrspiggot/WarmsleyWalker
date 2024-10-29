from typing import List, Dict
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    size: int
    concurrent_tasks: int
    priority: int
    context_sharing: bool = True


SECTION_CONFIGS = {
    "MANAGER INFORMATION": BatchConfig(
        size=20,  # Larger batches for simple questions
        concurrent_tasks=5,
        priority=1,
        context_sharing=True
    ),
    "FUND INFORMATION": BatchConfig(
        size=15,
        concurrent_tasks=4,
        priority=2,
        context_sharing=True
    ),
    "STRATEGY": BatchConfig(
        size=5,  # Smaller batches for complex questions
        concurrent_tasks=2,
        priority=3,
        context_sharing=False
    ),
    "DEFAULT": BatchConfig(
        size=10,
        concurrent_tasks=3,
        priority=2,
        context_sharing=True
    )
}


class BatchProcessor:
    def __init__(self, max_concurrent_total: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent_total)
        self.context_cache = {}

    async def process_items(self,
                            items: List[Dict],
                            section: str,
                            processor_func) -> List[Dict]:
        """Process items in optimized batches"""
        config = SECTION_CONFIGS.get(section.upper(), SECTION_CONFIGS["DEFAULT"])

        # Split items into batches
        batches = [items[i:i + config.size]
                   for i in range(0, len(items), config.size)]

        # Process batches with controlled concurrency
        results = []
        async with asyncio.TaskGroup() as group:
            batch_tasks = []
            for batch in batches:
                task = group.create_task(
                    self._process_batch(
                        batch=batch,
                        config=config,
                        processor_func=processor_func
                    )
                )
                batch_tasks.append(task)

            # Collect results maintaining order
            for task in batch_tasks:
                batch_results = await task
                results.extend(batch_results)

        return results

    async def _process_batch(self, batch: List[Dict], config: BatchConfig, processor_func) -> List[Dict]:
        """Process a single batch of items"""
        results = []
        semaphore = asyncio.Semaphore(config.concurrent_tasks)

        async def process_with_semaphore(item):
            async with self.semaphore, semaphore:
                try:
                    result = await processor_func(item)
                    logger.info(f"Successfully processed item {getattr(item, 'id', 'unknown')}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing item: {str(e)}")
                    return None

        # Start all tasks in the batch
        async with asyncio.TaskGroup() as group:
            tasks = []
            for item in batch:
                task = group.create_task(process_with_semaphore(item))
                tasks.append(task)

            # Collect results, filtering out None values from errors
            results = [task.result() for task in tasks if task.result() is not None]
            logger.info(f"Batch completed with {len(results)} successful results")

        return results


