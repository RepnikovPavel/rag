import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node, Query
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Глобальная инициализация языка (тяжелый объект)
PY_LANGUAGE = Language(tspython.language())

@dataclass
class CodeChunk:
    """Контейнер для единицы извлеченного кода."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParseResult:
    """Результат парсинга одного файла."""
    chunks: List[CodeChunk]
    file_imports: List[str]

def get_parser() -> Parser:
    """Создает и возвращает экземпляр парсера."""
    return Parser(PY_LANGUAGE)

def get_node_text(node: Node, source_bytes: bytes) -> str:
    """Безопасное извлечение текста узла."""
    return source_bytes[node.start_byte:node.end_byte].decode('utf8')

def extract_imports_nodes(root_node: Node, source_bytes: bytes) -> List[str]:
    """
    Извлекает текст всех импортов из файла используя query.matches для совместимости.
    """
    # S-выражение для поиска импортов
    query_pattern = "(import_statement) @import (import_from_statement) @import"
    query = Query(PY_LANGUAGE, query_pattern)
    
    # Используем matches() вместо captures() для максимальной совместимости версий
    matches = query.matches(root_node)
    
    imports = []
    seen_imports = set()

    for match in matches:
        # match.captures возвращает список кортежей: (index_of_capture_name, node)
        for capture in match.captures:
            node = capture[1]
            text = get_node_text(node, source_bytes)
            
            if text not in seen_imports:
                imports.append(text)
                seen_imports.add(text)
            
    return imports

def match_imports_to_chunk(chunk_text: str, file_imports: List[str]) -> str:
    """
    Сопоставляет импорты с кодом чанка.
    """
    relevant_imports = []
    # Список слов, которые нужно игнорировать при поиске (ключевые слова)
    keywords = {'import', 'from', 'as', 'class', 'def', 'return', 'if', 'else', 'elif', 'try', 'except', 'with'}
    
    for imp_line in file_imports:
        # Разбиваем строку импорта на токены
        tokens = [t for t in imp_line.split() if t not in keywords and not t.startswith('.')]
        
        found = False
        for token in tokens:
            # Очищаем токен от знаков препинания (запятые, точки в конце модулей)
            clean_token = token.strip(',.')
            
            # Проверяем вхождение токена в текст чанка
            if clean_token in chunk_text:
                found = True
                break
        
        if found:
            relevant_imports.append(imp_line)
            
    return "\n".join(relevant_imports)

def traverse_and_extract(
    node: Node, 
    source_bytes: bytes, 
    file_imports: List[str], 
    parent_class: Optional[str] = None
) -> List[CodeChunk]:
    """Рекурсивно обходит узлы AST."""
    chunks = []
    
    if node.type == "class_definition":
        # Извлекаем весь класс
        class_text = get_node_text(node, source_bytes)
        class_imports = match_imports_to_chunk(class_text, file_imports)
        
        final_text = f"{class_imports}\n\n{class_text}".strip()
        
        # Получаем имя класса
        class_name_node = node.child_by_field_name("name")
        class_name = get_node_text(class_name_node, source_bytes) if class_name_node else "UnknownClass"
        
        meta = {
            "type": "class",
            "name": class_name,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "parent_class": parent_class
        }
        
        chunks.append(CodeChunk(text=final_text, metadata=meta))
        
        # Рекурсия внутри класса
        for child in node.children:
            chunks.extend(traverse_and_extract(child, source_bytes, file_imports, parent_class=class_name))
            
    elif node.type == "function_definition":
        # Извлекаем функцию
        func_text = get_node_text(node, source_bytes)
        func_imports = match_imports_to_chunk(func_text, file_imports)
        
        final_text = f"{func_imports}\n\n{func_text}".strip()
        
        func_name_node = node.child_by_field_name("name")
        func_name = get_node_text(func_name_node, source_bytes) if func_name_node else "UnknownFunction"
        
        meta = {
            "type": "function",
            "name": func_name,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "parent_class": parent_class
        }
        
        chunks.append(CodeChunk(text=final_text, metadata=meta))
        
    else:
        # Рекурсия по остальным узлам
        for child in node.children:
            chunks.extend(traverse_and_extract(child, source_bytes, file_imports, parent_class))
            
    return chunks

def parse_python_file(source_code: str) -> ParseResult:
    """Главная точка входа."""
    parser = get_parser()
    source_bytes = bytes(source_code, "utf8")
    tree = parser.parse(source_bytes)
    root_node = tree.root_node
    
    # 1. Извлекаем импорты
    imports = extract_imports_nodes(root_node, source_bytes)
    
    # 2. Извлекаем чанки
    chunks = traverse_and_extract(root_node, source_bytes, imports)
    
    return ParseResult(chunks=chunks, file_imports=imports)

# --- Тестирование ---
if __name__ == "__main__":
    sample_code = """
import os
import sys
import pandas as pd
from typing import List

class DataProcessor:
    '''Класс для обработки данных'''
    
    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> pd.DataFrame:
        '''Загружает данные'''
        if not os.path.exists(self.path):
            raise FileNotFoundError
        return pd.read_csv(self.path)

def helper_function():
    print(sys.version)
"""

    result = parse_python_file(sample_code)

    print(f"Found {len(result.chunks)} chunks.")
    print("-" * 40)
    
    for chunk in result.chunks:
        print(f"Type: {chunk.metadata['type']} | Name: {chunk.metadata.get('name')}")
        print(chunk.text[:200] + "...")
        print()