import json

def add_service_to_json_file(file_path, new_service):
    """
    从JSON文件读取服务数据，添加新服务后写回文件
    
    Args:
        file_path (str): JSON文件路径（例如 "services.json"）
        new_service (str): 要添加的服务名称（如 "checkoutservice2-0"）
    """
    # 1. 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 检查并添加服务（避免重复）
    for group in data.values():
        if new_service not in group:
            group.append(new_service)
    
    # 3. 写回文件（可选格式化缩进）
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"成功添加服务 '{new_service}' 到文件 {file_path}")

# 使用示例
if __name__ == "__main__":
    add_service_to_json_file("nodes.json", "currencyservice-2")