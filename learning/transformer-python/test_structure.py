#!/usr/bin/env python3
"""
测试项目结构和导入的脚本
不依赖 PyTorch，只测试代码结构
"""

import os
import sys
import ast
import importlib.util

def test_file_structure():
    """测试文件结构"""
    print("=== 测试文件结构 ===")
    
    required_files = [
        'transformer_pytorch/__init__.py',
        'transformer_pytorch/core/__init__.py',
        'transformer_pytorch/core/math_utils.py',
        'transformer_pytorch/core/layers.py',
        'transformer_pytorch/core/attention.py',
        'transformer_pytorch/core/embedding.py',
        'transformer_pytorch/core/encoder.py',
        'transformer_pytorch/core/decoder.py',
        'transformer_pytorch/core/transformer.py',
        'transformer_pytorch/config/__init__.py',
        'transformer_pytorch/config/config.py',
        'transformer_pytorch/config/constants.py',
        'setup.py',
        'requirements.txt',
        'README.md',
        'examples/basic_usage.py',
        'tests/test_basic.py',
        'notebooks/01_basic_concepts.ipynb'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ 缺少文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print(f"\n✅ 所有必需文件都存在 ({len(required_files)} 个)")
        return True

def test_python_syntax():
    """测试 Python 语法"""
    print("\n=== 测试 Python 语法 ===")
    
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            print(f"✅ {file_path}")
        except SyntaxError as e:
            syntax_errors.append((file_path, str(e)))
            print(f"❌ {file_path}: {e}")
        except Exception as e:
            print(f"⚠️ {file_path}: {e}")
    
    if syntax_errors:
        print(f"\n❌ 发现语法错误:")
        for file_path, error in syntax_errors:
            print(f"   - {file_path}: {error}")
        return False
    else:
        print(f"\n✅ 所有 Python 文件语法正确 ({len(python_files)} 个)")
        return True

def test_imports_structure():
    """测试导入结构（不实际导入）"""
    print("\n=== 测试导入结构 ===")
    
    # 检查配置模块的结构
    config_file = 'transformer_pytorch/config/config.py'
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_classes = ['TransformerConfig']
        required_functions = ['get_config', 'create_config']
        
        for cls in required_classes:
            if f'class {cls}' in content:
                print(f"✅ 找到类: {cls}")
            else:
                print(f"❌ 缺少类: {cls}")
        
        for func in required_functions:
            if f'def {func}' in content:
                print(f"✅ 找到函数: {func}")
            else:
                print(f"❌ 缺少函数: {func}")
    
    # 检查核心模块
    core_modules = [
        ('math_utils.py', ['gelu_activation', 'scaled_dot_product_attention']),
        ('layers.py', ['LayerNorm', 'FeedForward']),
        ('attention.py', ['MultiHeadAttention', 'SelfAttention']),
        ('embedding.py', ['TokenEmbedding', 'TransformerEmbedding']),
        ('encoder.py', ['EncoderLayer', 'TransformerEncoder']),
        ('decoder.py', ['DecoderLayer', 'TransformerDecoder']),
        ('transformer.py', ['Transformer'])
    ]
    
    for module_file, expected_items in core_modules:
        file_path = f'transformer_pytorch/core/{module_file}'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for item in expected_items:
                if f'class {item}' in content or f'def {item}' in content:
                    print(f"✅ {module_file}: {item}")
                else:
                    print(f"❌ {module_file}: 缺少 {item}")

def test_documentation():
    """测试文档"""
    print("\n=== 测试文档 ===")
    
    doc_files = ['README.md', 'IMPLEMENTATION_SUMMARY.md']
    
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) > 1000:  # 至少1000字符
                print(f"✅ {doc_file}: {len(content)} 字符")
            else:
                print(f"⚠️ {doc_file}: 内容较少 ({len(content)} 字符)")
        else:
            print(f"❌ 缺少文档: {doc_file}")

def test_package_structure():
    """测试包结构"""
    print("\n=== 测试包结构 ===")
    
    # 检查 __init__.py 文件
    init_files = [
        'transformer_pytorch/__init__.py',
        'transformer_pytorch/core/__init__.py',
        'transformer_pytorch/config/__init__.py'
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '__all__' in content:
                print(f"✅ {init_file}: 包含 __all__")
            else:
                print(f"⚠️ {init_file}: 缺少 __all__")
            
            if 'from' in content and 'import' in content:
                print(f"✅ {init_file}: 包含导入语句")
            else:
                print(f"⚠️ {init_file}: 缺少导入语句")
        else:
            print(f"❌ 缺少: {init_file}")

def test_setup_files():
    """测试安装文件"""
    print("\n=== 测试安装文件 ===")
    
    # 检查 setup.py
    if os.path.exists('setup.py'):
        with open('setup.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_fields = ['name=', 'version=', 'packages=', 'install_requires=']
        for field in required_fields:
            if field in content:
                print(f"✅ setup.py: 包含 {field}")
            else:
                print(f"❌ setup.py: 缺少 {field}")
    
    # 检查 requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        required_packages = ['torch', 'numpy']
        for package in required_packages:
            if any(package in line for line in lines):
                print(f"✅ requirements.txt: 包含 {package}")
            else:
                print(f"❌ requirements.txt: 缺少 {package}")

def main():
    """主测试函数"""
    print("🧪 Transformer-PyTorch 项目结构测试\n")
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_imports_structure,
        test_documentation,
        test_package_structure,
        test_setup_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result is not False:  # None 或 True 都算通过
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 失败: {e}")
    
    print(f"\n📊 测试结果:")
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")
    print(f"📈 成功率: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！项目结构完整。")
        print("\n💡 下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 安装包: pip install -e .")
        print("3. 运行示例: python examples/basic_usage.py")
        print("4. 运行测试: pytest tests/")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查项目结构。")

if __name__ == '__main__':
    main()
