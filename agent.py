"""vLLM model loading, generation, and resource cleanup."""

import math
import multiprocessing
import os

import vllm
from transformers import AutoTokenizer


class VllmAgent:
    def __init__(self, model_name, model_kwargs, generation_kwargs):
        # handle case when rank and data_parallel_size are both specified
        assert not (("rank" in model_kwargs) and ("data_parallel_size" in model_kwargs)), "Both rank and data_parallel_size cannot be specified in model_kwargs."

        if ("data_parallel_size" in model_kwargs) and (model_kwargs["data_parallel_size"] > 1):
            self.data_parallel_size = model_kwargs["data_parallel_size"]
            model_kwargs.pop("data_parallel_size")
            self.llm = None
        else:
            if "data_parallel_size" in model_kwargs:
                model_kwargs.pop("data_parallel_size")
            self.data_parallel_size = 1
            if "rank" in model_kwargs:
                self.rank = model_kwargs["rank"]
                model_kwargs.pop("rank")
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.rank)
            self.llm = vllm.LLM(
                model=model_name,
                tokenizer=model_name,
                **model_kwargs
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.model_kwargs = model_kwargs

        self.sampling_params = vllm.SamplingParams(
            **generation_kwargs
        )

    def shutdown(self):
        if self.llm is not None:
            llm = self.llm
            self.llm = None

            # Release vLLM engine resources.
            engine = getattr(llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                engine.shutdown()

            del llm

    def _generation_worker_main(self, rank, model_name, model_kwargs, sampling_params, prompts):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        llm = vllm.LLM(
            model=model_name,
            **model_kwargs
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=(rank==0))
        return outputs

    def generate(self, prompt):
        if self.data_parallel_size > 1:
            world_size = self.data_parallel_size
            print(f"Using {world_size} GPUs for generation.")
            per_device_samples = math.ceil(len(prompt) / world_size)
            args = [(rank, self.model_name, self.model_kwargs, self.sampling_params, prompt[rank * per_device_samples: (rank + 1) * per_device_samples]) for rank in range(world_size)]
            with multiprocessing.Pool(world_size) as pool:
                results = pool.starmap(self._generation_worker_main, args)
            outputs = []
            for result in results:
                outputs.extend(result)
        else:
            llm = self.llm
            outputs = llm.generate(prompt, self.sampling_params)

        prompt_to_output = {
                g.prompt: [g.outputs[i].text for i in range(len(g.outputs))] for g in outputs
            }
        outputs = [prompt_to_output[p] if p in prompt_to_output else "" for p in prompt]

        return outputs
