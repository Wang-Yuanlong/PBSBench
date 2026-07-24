import copy

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


class CaptionCollator:
    def __call__(self, batch):
        output = {"images": torch.stack([item["images"] for item in batch]),
                  "descriptions": [item["descriptions"] for item in batch]}
        if "patches" in batch[0]:
            output["patches"] = torch.stack([item["patches"] for item in batch])
        return output


class QwenQACollator:
    """Build Qwen chat inputs with a fixed number of PBS visual tokens."""

    template = [
        {"role": "user", "content": [{"type": "image", "image": None},
                                     {"type": "text", "text": None}]},
        {"role": "assistant", "content": [{"type": "text", "text": None}]},
    ]

    def __init__(self, processor, num_query_tokens=32, slide=False, include_answers=True):
        self.processor = processor
        self.num_query_tokens = num_query_tokens
        self.slide = slide
        self.include_answers = include_answers

    def _format(self, item):
        question, answer = item["question"], str(item["answer"])
        kind, options = item.get("question_type", "open"), item.get("options")
        if kind == "multiple_choice" and options:
            labels = [chr(ord("A") + index) for index in range(len(options))]
            rendered = [f"{label}. {option}" for label, option in zip(labels, options)]
            if answer in options:
                answer = labels[options.index(answer)]
            question += "\nOptions:\n" + "\n".join(rendered)
        elif kind == "true_false":
            question += "\nAnswer with only true or false."
            answer = answer.lower()
        elif kind == "fill_blank":
            question += "\nAnswer in no more than 10 words."
        return question, answer

    def _replace_image_tokens(self, ids, mask):
        positions = (ids == self.processor.image_token_id).nonzero(as_tuple=True)[1]
        if not len(positions):
            raise ValueError("chat template produced no image tokens")
        start, end = positions[0], positions[-1]
        tokens = torch.full((1, self.num_query_tokens), self.processor.image_token_id,
                            dtype=ids.dtype)
        ones = torch.ones((1, self.num_query_tokens), dtype=mask.dtype)
        return (torch.cat((ids[:, :start], tokens, ids[:, end + 1:]), 1),
                torch.cat((mask[:, :start], ones, mask[:, end + 1:]), 1))

    def __call__(self, batch):
        items = []
        for record in batch:
            question, answer = self._format(record)
            message = copy.deepcopy(self.template)
            message[0]["content"][0]["image"] = Image.new("RGB", (224, 224)) if self.slide else to_pil_image(record["image"])
            message[0]["content"][1]["text"] = question
            if self.include_answers:
                message[1]["content"][0]["text"] = answer
            else:
                message = message[:1]
            encoded = self.processor.apply_chat_template(
                message, add_generation_prompt=not self.include_answers, tokenize=True,
                return_dict=True, return_tensors="pt")
            encoded["input_ids"], encoded["attention_mask"] = self._replace_image_tokens(
                encoded["input_ids"], encoded["attention_mask"])
            if self.include_answers:
                prompt = self.processor.apply_chat_template(
                    message[:1], add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                prompt["input_ids"], prompt["attention_mask"] = self._replace_image_tokens(
                    prompt["input_ids"], prompt["attention_mask"]
                )
                encoded["labels"] = encoded["input_ids"].clone()
                encoded["labels"][0, :prompt["input_ids"].shape[1]] = -100
            encoded["pixel_values"] = record["image"]
            encoded["image_grid_thw"] = torch.tensor([[1, 1, self.num_query_tokens]])
            items.append(encoded)

        length = max(item["input_ids"].shape[1] for item in items)
        result = {}
        keys = ("input_ids", "attention_mask", "labels") if self.include_answers else (
            "input_ids", "attention_mask"
        )
        for key in keys:
            value = -100 if key == "labels" else (0 if key == "attention_mask" else self.processor.tokenizer.pad_token_id)
            padded = []
            for item in items:
                amount = length - item[key].shape[1]
                padding = (0, amount) if self.include_answers else (amount, 0)
                padded.append(torch.nn.functional.pad(item[key], padding, value=value))
            result[key] = torch.cat(padded)
        if self.slide:
            tokens = max(item["pixel_values"].shape[0] for item in items)
            padded = []
            for item in items:
                features = item["pixel_values"]
                amount = tokens - features.shape[0]
                padding = (0, 0, 0, amount) if features.ndim == 2 else (
                    0, 0, 0, 0, 0, amount
                )
                padded.append(torch.nn.functional.pad(features, padding))
            result["pixel_values"] = torch.stack(padded)
        else:
            result["pixel_values"] = torch.stack([item["pixel_values"] for item in items])
        result["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in items])
        return result
