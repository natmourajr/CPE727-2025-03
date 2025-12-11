from __future__ import annotations
from typing import List, Dict, Any
import copy


class CategoryHarmonizer:
    """
    Unifies categories and remaps IDs across multiple experiments.
    """

    @staticmethod
    def unify(parsed_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 1. Build GLOBAL category table
        # ------------------------------------------------------------------
        global_categories: Dict[str, int] = {}
        next_cat_id = 0

        for dataset in parsed_list:
            for cat in dataset.get("categories", []):
                name = cat["name"]
                if name not in global_categories:
                    global_categories[name] = next_cat_id
                    next_cat_id += 1

        # Convert dict → ordered list
        final_categories = [
            {"id": cid, "name": name}
            for name, cid in global_categories.items()
        ]

        # ------------------------------------------------------------------
        # 2. Merge datasets with ID offsets
        # ------------------------------------------------------------------
        global_images = []
        global_annotations = []

        img_offset = 0
        ann_offset = 0

        for dataset in parsed_list:
            images = dataset.get("images", [])
            annotations = dataset.get("annotations", [])
            local_categories = dataset.get("categories", [])

            # Images
            for img in images:
                new_img = copy.deepcopy(img)
                new_img["id"] = img["id"] + img_offset
                global_images.append(new_img)

            # Annotations
            for ann in annotations:
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = ann["id"] + ann_offset
                new_ann["image_id"] = ann["image_id"] + img_offset

                # Remap category by name
                old_cat_name = local_categories[ann["category_id"]]["name"]
                new_ann["category_id"] = global_categories[old_cat_name]

                global_annotations.append(new_ann)

            img_offset += len(images)
            ann_offset += len(annotations)

        # Final dataset before ID shift
        dataset = {
            "images": global_images,
            "annotations": global_annotations,
            "categories": final_categories
        }
        
        return dataset
