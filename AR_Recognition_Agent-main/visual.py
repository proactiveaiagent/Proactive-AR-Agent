from src.memory.chroma_memory import ChromaUserMemory
from src.utils.chroma_visualizer import ChromaVisualizer, visualize_memory
from src.cli import build_memory
import os

# 构建 memory
memory = build_memory('./chroma_db', '/userhome/cs/u3651279/data/emb/all-MiniLM-L6-v2')

# 方式 1: 使用便捷函数
visualize_memory(memory, "test_user")

# 方式 2: 详细使用
viz = ChromaVisualizer(memory)

# 1. 显示完整摘要
viz.print_summary("test_user")

# 2. 获取所有物体数据框
df = viz.get_all_objects("test_user")
if not df.empty:
    print(df.head())

# 3. 查看原始数据（调试用）
raw = viz.get_raw_objects("test_user", limit=5)
print("\nRaw data structure:")
print(raw[0] if raw else "No data")

# 4. 获取统计信息
stats = viz.get_statistics("test_user")
print(f"\nStatistics: {stats}")

# 5. 获取类型分布
type_dist = viz.get_type_distribution("test_user")
print("\nType Distribution:")
print(type_dist)

# 6. 获取名称分布
name_dist = viz.get_name_distribution("test_user")
print("\nName Distribution:")
print(name_dist.head(10))

# 7. 搜索特定类型
mugs = viz.search_by_type("test_user", "mug")
print(f"\nFound {len(mugs)} mugs")

# 8. 搜索特定名称
coffee_items = viz.search_by_name("test_user", "coffee")
print(f"\nFound {len(coffee_items)} items with 'coffee' in name")

# 9. 获取最常见的物体
top_objects = viz.get_most_seen_objects("test_user", top_n=5)
print("\nTop 5 Most Seen:")
print(top_objects)

# 10. 获取最近的物体
recent = viz.get_recently_seen("test_user", top_n=5)
print("\nRecently Seen:")
print(recent)

# 11. 导出数据
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)
viz.export_to_csv("test_user", "./outputs/memory_export.csv")