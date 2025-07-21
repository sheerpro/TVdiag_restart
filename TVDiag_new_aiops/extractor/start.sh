#!/bin/bash
# 从下到上依次执行文件（按图片中显示的顺序）

# 1. 首先执行最下方的文件
python raw_process.py

# 2. 执行倒数第二个文件
python preprocess.py

# 3. 执行中间的日志模板提取文件（注意文件名可能需要调整）
python log_template_extractor.py  # 请根据实际文件名修改

# 4. 执行第二个文件
python event_extractor.py        # 图片中显示为"event éxito rabbit"，可能需要确认实际文件名

# 5. 最后执行最上方的文件
python deployment_extractor.py   # 图片中显示为"depoyment_factor.py"，可能需要确认实际文件名

echo "所有脚本已按顺序执行完成"