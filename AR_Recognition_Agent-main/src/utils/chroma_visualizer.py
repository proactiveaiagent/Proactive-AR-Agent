from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from ..memory.chroma_memory import ChromaUserMemory


class ChromaVisualizer:
    """ChromaDB Visualization and Analysis Tool"""
    
    def __init__(self, memory: ChromaUserMemory):
        self.memory = memory
    
    def get_all_objects(self, user_id: str, limit: int = 1000) -> pd.DataFrame:
        """
        Get all stored objects for a user as a DataFrame.
        Args:
            user_id: User ID
            limit: Maximum number of objects to retrieve
        Returns:
            DataFrame of objects
        """
        # query all objects
        results = self.memory.query(
            user_id=user_id,
            query="object",
            k=limit,
            filter_metadata={"type": "object"}
        )
        
        if not results:
            print("No objects found in memory.")
            return pd.DataFrame()
        
        # extract object data into DataFrame
        objects_data = []
        for result in results:
            # 修复：从 event 中获取 object_data
            event = result.get("event", {})
            obj_data = event.get("object_data", {})
            
            if not obj_data:
                continue
            
            # ensure all expected fields are present
            objects_data.append({
                "object_id": obj_data.get("object_id", "N/A"),
                "object_name": obj_data.get("object_name", "N/A"),
                "object_type": obj_data.get("object_type", "N/A"),
                "spatial_relation": obj_data.get("spatial_relation", "N/A"),
                "current_state": obj_data.get("current_state", "N/A"),
                "affordance": ", ".join(obj_data.get("affordance", [])),
                "digital_connectivity": obj_data.get("digital_connectivity", "N/A"),
                "first_seen": obj_data.get("first_seen", "N/A"),
                "last_seen": obj_data.get("last_seen", "N/A"),
                "seen_count": obj_data.get("seen_count", 0),
                "similarity": result.get("similarity", 0.0),
                "distance": result.get("distance", 0.0)
            })
        
        df = pd.DataFrame(objects_data)
        
        # set pandas display options for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', 50)
        pd.set_option('display.width', None)
        
        return df
    
    def get_object_detail(self, user_id: str, object_id: str) -> pd.Series:
        """
        Get detailed information of a specific object by its ID.
        Args:
            user_id: User ID
            object_id: Object ID
        Returns:
            Series with object details
        """
        results = self.memory.query(
            user_id=user_id,
            query=object_id,
            k=10,  # 增加 k 值以确保找到
            filter_metadata={"type": "object", "object_id": object_id}
        )
        
        if not results:
            print(f"Object {object_id} not found.")
            return pd.Series()
        
        # 修复：从 event 中获取 object_data
        event = results[0].get("event", {})
        obj_data = event.get("object_data", {})
        
        if not obj_data:
            print(f"Object data for {object_id} is empty.")
            return pd.Series()
        
        return pd.Series(obj_data)
    
    def get_raw_objects(self, user_id: str, limit: int = 1000) -> list[dict]:
        """
        Get raw object data (for debugging).
        
        Args:
            user_id: User ID
            limit: Maximum number of objects to retrieve
            
        Returns:
            List of raw result dictionaries
        """
        results = self.memory.query(
            user_id=user_id,
            query="object",
            k=limit,
            filter_metadata={"type": "object"}
        )
        return results
    
    def get_statistics(self, user_id: str) -> dict:
        """
        Get overall statistics of the user's memory.
        Args:
            user_id: User ID
        Returns:
            Dictionary of statistics
        """
        df = self.get_all_objects(user_id, limit=10000)
        
        if df.empty:
            return {}
        
        stats = {
            "total_objects": len(df),
            "total_appearances": int(df["seen_count"].sum()),
            "avg_appearances": round(df["seen_count"].mean(), 2),
            "max_appearances": int(df["seen_count"].max()),
            "unique_types": df["object_type"].nunique(),
            "unique_names": df["object_name"].nunique(),
        }
        
        return stats
    
    def get_type_distribution(self, user_id: str) -> pd.DataFrame:
        """
        Get distribution of object types.
        Args:
            user_id: User ID
        Returns:
            DataFrame with type distribution
        """
        df = self.get_all_objects(user_id, limit=10000)
        
        if df.empty:
            return pd.DataFrame()
        
        type_dist = df.groupby("object_type").agg({
            "object_id": "count",
            "seen_count": ["sum", "mean", "max"]
        }).round(2)
        
        type_dist.columns = ["count", "total_seen", "avg_seen", "max_seen"]
        type_dist = type_dist.sort_values("count", ascending=False)
        
        return type_dist
    
    def get_name_distribution(self, user_id: str) -> pd.DataFrame:
        """
        Get distribution of object names.
        Args:
            user_id: User ID
        Returns:
            DataFrame with name distribution
        """
        df = self.get_all_objects(user_id, limit=10000)
        
        if df.empty:
            return pd.DataFrame()
        
        name_dist = df.groupby("object_name").agg({
            "object_id": "count",
            "object_type": "first",
            "seen_count": ["sum", "mean", "max"]
        }).round(2)
        
        name_dist.columns = ["count", "type", "total_seen", "avg_seen", "max_seen"]
        name_dist = name_dist.sort_values("total_seen", ascending=False)
        
        return name_dist
    
    def search_by_type(self, user_id: str, object_type: str) -> pd.DataFrame:
        """
        Search objects by type.
        Args:
            user_id: User ID
            object_type: Object type to search
        Returns:
            DataFrame of matched objects
        """
        df = self.get_all_objects(user_id, limit=10000)
        
        if df.empty:
            return pd.DataFrame()
        
        # Don't care about case
        filtered = df[df["object_type"].str.lower() == object_type.lower()]
        
        return filtered.sort_values("seen_count", ascending=False)
    
    def search_by_name(self, user_id: str, name_pattern: str) -> pd.DataFrame:
        """
        Search objects by name pattern (fuzzy search).
        Args:
            user_id: User ID
            name_pattern: Name pattern (substring)
        Returns:
            DataFrame of matched objects
        """
        df = self.get_all_objects(user_id, limit=10000)
        
        if df.empty:
            return pd.DataFrame()
        
        # 不区分大小写模糊搜索
        filtered = df[df["object_name"].str.contains(name_pattern, case=False, na=False)]
        
        return filtered.sort_values("seen_count", ascending=False)
    
    def get_most_seen_objects(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get most frequently seen objects.
        
        Args:
            user_id: User ID
            top_n: Number of top objects to return
            
        Returns:
            DataFrame
        """
        df = self.get_all_objects(user_id, limit=10000)
        
        if df.empty:
            return pd.DataFrame()
        
        return df.nlargest(top_n, "seen_count")[
            ["object_name", "object_type", "seen_count", "first_seen", "last_seen"]
        ]
    
    def get_recently_seen(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get recently seen objects.
        
        Args:
            user_id: User ID
            top_n: Number of recent objects to return
            
        Returns:
            DataFrame
        """
        df = self.get_all_objects(user_id, limit=10000)
        
        if df.empty:
            return pd.DataFrame()
        
        # transform last_seen to datetime for sorting
        df["last_seen_dt"] = pd.to_datetime(df["last_seen"], errors="coerce")
        df_sorted = df.sort_values("last_seen_dt", ascending=False)
        
        return df_sorted.head(top_n)[
            ["object_name", "object_type", "seen_count", "first_seen", "last_seen"]
        ]
    
    def export_to_csv(self, user_id: str, filepath: str) -> None:
        """
        Export memory data to a CSV file.
        
        Args:
            user_id: User ID
            filepath: output CSV file path
        """
        df = self.get_all_objects(user_id, limit=100000)
        
        if not df.empty:
            df.to_csv(filepath, index=False, encoding="utf-8")
            print(f"✓ Exported {len(df)} objects to {filepath}")
        else:
            print("No data to export.")
    
    def print_summary(self, user_id: str) -> None:
        """
        Print a summary of the user's memory.
        
        Args:
            user_id: User ID
        """
        print(f"\n{'='*80}")
        print(f"ChromaDB Memory Summary for User: {user_id}")
        print(f"{'='*80}\n")
        
        stats = self.get_statistics(user_id)
        if stats:
            print("📊 Overall Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        else:
            print("⚠️  No statistics available (empty memory).\n")
            return
        
        print("📦 Object Type Distribution:")
        type_dist = self.get_type_distribution(user_id)
        if not type_dist.empty:
            print(type_dist.to_string())
            print()
        else:
            print("  No type distribution available.\n")
        
        print("🏷️  Object Name Distribution (Top 10):")
        name_dist = self.get_name_distribution(user_id)
        if not name_dist.empty:
            print(name_dist.head(10).to_string())
            print()
        else:
            print("  No name distribution available.\n")
        
        print("🔝 Top 10 Most Seen Objects:")
        top_seen = self.get_most_seen_objects(user_id, top_n=10)
        if not top_seen.empty:
            print(top_seen.to_string(index=False))
            print()
        else:
            print("  No objects found.\n")

        print("🕒 Recently Seen Objects:")
        recent = self.get_recently_seen(user_id, top_n=10)
        if not recent.empty:
            print(recent.to_string(index=False))
        else:
            print("  No objects found.\n")
        
        print(f"\n{'='*80}\n")


def visualize_memory(memory: ChromaUserMemory, user_id: str) -> None:
    viz = ChromaVisualizer(memory)
    viz.print_summary(user_id)