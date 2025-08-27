from pymongo import MongoClient
from pprint import pprint
from src.infrastructure.config import MONGODB_URI


def gather_stats(col, filter_query=None):
    _filter_query = filter_query or {}

    sample = col.find_one(_filter_query)
    if not sample:
        return {}

    fields = [f for f in sample.keys() if f not in ["_id", "created_at", "updated_at"]]

    group_stage = {"_id": None, "count": {"$sum": 1}}
    numeric_fields, text_fields = [], []

    for f in fields:
        val = sample[f]
        if isinstance(val, (int, float)):
            numeric_fields.append(f)
            group_stage[f"{f}_min"] = {"$min": f"${f}"}
            group_stage[f"{f}_max"] = {"$max": f"${f}"}
            group_stage[f"{f}_avg"] = {"$avg": f"${f}"}
            group_stage[f"{f}_sum"] = {"$sum": f"${f}"}
            group_stage[f"{f}_stddev"] = {"$stdDevPop": f"${f}"}
            group_stage[f"{f}_null_count"] = {
                "$sum": {"$cond": [{"$eq": [f"${f}", None]}, 1, 0]}
            }
        elif isinstance(val, str):
            text_fields.append(f)
            group_stage[f"{f}_null_count"] = {
                "$sum": {"$cond": [{"$eq": [f"${f}", None]}, 1, 0]}
            }

    agg_result = list(
        col.aggregate([{"$match": _filter_query}, {"$group": group_stage}])
    )
    raw = agg_result[0] if agg_result else {}

    # Мода для числовых полей
    modes = {}
    for f in numeric_fields:
        mode_pipeline = [
            {"$match": {"$and": [{f: {"$ne": None}}, _filter_query]}},
            {"$group": {"_id": f"${f}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1},
        ]
        mode_result = list(col.aggregate(mode_pipeline))
        modes[f] = mode_result[0]["_id"] if mode_result else None

    # Перцентили для числовых
    percentiles_stage = {
        "$group": {
            "_id": None,
            **{
                f"{f}_percentiles": {
                    "$percentile": {
                        "input": f"${f}",
                        "p": [0.25, 0.5, 0.75],
                        "method": "approximate",
                    }
                }
                for f in numeric_fields
            },
        }
    }
    perc_raw = list(col.aggregate([{"$match": _filter_query}, percentiles_stage]))[0]

    # Формируем человекочитаемый результат
    result = {"count": raw["count"], "fields": {}}

    for f in numeric_fields:
        q1, median, q3 = None, None, None
        if f"{f}_percentiles" in perc_raw:
            vals = perc_raw[f"{f}_percentiles"]
            if vals:
                q1, median, q3 = vals[0], vals[1], vals[2]

        result["fields"][f] = {
            "min": raw.get(f"{f}_min"),
            "max": raw.get(f"{f}_max"),
            "range": (raw.get(f"{f}_max") - raw.get(f"{f}_min"))
            if raw.get(f"{f}_min") is not None
            else None,
            "avg": raw.get(f"{f}_avg"),
            "sum": raw.get(f"{f}_sum"),
            "stddev": raw.get(f"{f}_stddev"),
            "q1": q1,
            "median": median,
            "q3": q3,
            "null_count": raw.get(f"{f}_null_count"),
            "mode": modes[f],
        }

    for f in text_fields:
        result["fields"][f] = {
            "count": raw["count"],
            "null_count": raw.get(f"{f}_null_count"),
        }

    return result


if __name__ == "__main__":
    client = MongoClient(MONGODB_URI)
    db = client["db"]
    collection = db[
        "Amtliche_Korrekturen_und_reguläre_Wanderungsvorgänge_68ae365ac09727dd4725c9d0"
    ]

    # пример фильтра: даты или числовые значения
    filters = {
        # "some_numeric_field": {"$gte": 0},
        "Jahr": {"$gte": 2022},
    }

    stats = gather_stats(collection, filters)
    pprint(stats)
