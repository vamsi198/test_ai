import ast
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from config import Config


class DocumentationGenerator:
    """
    Generate comprehensive documentation from code analysis results
    Supports Markdown and HTML formats
    """
    
    def __init__(self):
        self.config = Config
        self.doc_format = self.config.DOC_FORMAT
        self.include_examples = self.config.INCLUDE_EXAMPLES
    
    
    def generate_full_documentation(
        self, 
        filepath: str, 
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete documentation for analyzed code
        """
        try:
            # Read code
            with open(filepath, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Get file info
            filename = os.path.basename(filepath)
            file_extension = os.path.splitext(filename)[1]
            
            # Generate documentation sections
            doc = {
                'filename': filename,
                'filepath': filepath,
                'language': self._detect_language(file_extension),
                'generated_at': datetime.now().isoformat(),
                'format': self.doc_format,
                'sections': {}
            }
            
            # 1. Overview
            doc['sections']['overview'] = self._generate_overview(
                filename, analysis_result
            )
            
            # 2. Module/File Description (IMPROVED FALLBACK)
            doc['sections']['description'] = self._generate_description(
                code_content, analysis_result
            )
            
            # 3. Functions Documentation (IMPROVED FALLBACK)
            if analysis_result.get('ast_analysis', {}).get('functions'):
                doc['sections']['functions'] = self._generate_functions_doc(
                    code_content, analysis_result['ast_analysis']['functions']
                )
            
            # 4. Classes Documentation
            if analysis_result.get('ast_analysis', {}).get('classes'):
                doc['sections']['classes'] = self._generate_classes_doc(
                    code_content, analysis_result['ast_analysis']['classes']
                )
            
            # 5. Dependencies/Imports
            if analysis_result.get('ast_analysis', {}).get('imports'):
                doc['sections']['dependencies'] = self._generate_dependencies_doc(
                    analysis_result['ast_analysis']['imports']
                )
            
            # 6. Code Quality Summary
            doc['sections']['quality_summary'] = self._generate_quality_summary(
                analysis_result
            )
            
            # 7. Generate formatted output
            if self.doc_format == 'markdown':
                doc['formatted_output'] = self._format_as_markdown(doc)
            elif self.doc_format == 'html':
                doc['formatted_output'] = self._format_as_html(doc)
            else:
                doc['formatted_output'] = self._format_as_markdown(doc)
            
            return doc
        
        except Exception as e:
            return {
                'error': str(e),
                'filename': os.path.basename(filepath) if filepath else 'unknown',
                'generated_at': datetime.now().isoformat()
            }
    
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension"""
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'React JSX',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript JSX',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        return language_map.get(extension, 'Unknown')
    
    
    def _generate_overview(
        self, 
        filename: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overview section"""
        summary = analysis.get('summary', {})
        
        return {
            'filename': filename,
            'total_lines': summary.get('total_lines', 0),
            'total_functions': summary.get('total_functions', 0),
            'total_classes': summary.get('total_classes', 0),
            'quality_score': summary.get('quality_score', 0),
            'quality_grade': summary.get('quality_grade', 'N/A')
        }
    
    
    def _generate_description( # <-- UPDATED METHOD
        self, 
        code: str, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate module/file description, providing comprehensive fallback."""
        ast_analysis = analysis.get('ast_analysis', {})
        module_docstring = ast_analysis.get('docstring')
        
        if module_docstring and module_docstring.strip():
            return module_docstring
        else:
            # Generate comprehensive basic description when no docstring exists
            summary = analysis.get('summary', {})
            functions = analysis.get('ast_analysis', {}).get('functions', [])
            classes = analysis.get('ast_analysis', {}).get('classes', [])
            imports = analysis.get('ast_analysis', {}).get('imports', [])

            description_parts = []
            description_parts.append(f"**Python module with {summary.get('total_lines', 0)} lines of code.**\n")

            if functions:
                func_names = [f['name'] for f in functions[:5]]
                description_parts.append(f"\n**Functions ({len(functions)} total):**")
                description_parts.append(f"- {', '.join(func_names)}")
                if len(functions) > 5:
                    description_parts.append(f"- ...and {len(functions) - 5} more")

            if classes:
                class_names = [c['name'] for c in classes[:3]]
                description_parts.append(f"\n**Classes ({len(classes)} total):**")
                description_parts.append(f"- {', '.join(class_names)}")
                if len(classes) > 3:
                    description_parts.append(f"- ...and {len(classes) - 3} more")

            if imports:
                description_parts.append(f"\n**Dependencies:** {len(imports)} imported module(s)")

            description_parts.append(f"\n**Code Quality:** {summary.get('quality_score', 0)}/100 (Grade: {summary.get('quality_grade', 'N/A')})")

            return '\n'.join(description_parts)
    
    
    def _generate_functions_doc(
        self, 
        code: str, 
        functions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate documentation for all functions, with smart docstring fallback."""
        functions_doc = []
        
        for func in functions:
            # Generate smart fallback if no docstring exists (FIX 2)
            docstring = func.get('docstring')
            if not docstring or not docstring.strip():
                params = func.get('args', [])
                if params:
                    docstring = f"Function with parameters: `{', '.join(params)}`\n\nNo detailed documentation provided."
                else:
                    docstring = "No parameters. No detailed documentation provided."
            
            func_doc = {
                'name': func['name'],
                'line_start': func['line_start'],
                'line_end': func['line_end'],
                'parameters': func.get('args', []),
                'decorators': func.get('decorators', []),
                'is_async': func.get('is_async', False),
                'docstring': docstring, # Use the smart fallback
                'signature': self._generate_function_signature(func)
            }
            
            functions_doc.append(func_doc)
        
        return functions_doc
    
    
    def _generate_classes_doc(
        self, 
        code: str, 
        classes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate documentation for all classes"""
        classes_doc = []
        
        for cls in classes:
            cls_doc = {
                'name': cls['name'],
                'line_start': cls['line_start'],
                'line_end': cls['line_end'],
                'methods': cls.get('methods', []),
                'base_classes': cls.get('bases', []),
                'docstring': cls.get('docstring', 'No docstring provided'),
                'method_count': len(cls.get('methods', []))
            }
            
            classes_doc.append(cls_doc)
        
        return classes_doc
    
    
    def _generate_dependencies_doc(
        self, 
        imports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate dependencies documentation"""
        standard_libs = set(['os', 'sys', 're', 'json', 'datetime', 'time', 'math', 
                             'random', 'collections', 'itertools', 'functools'])
        
        standard = []
        third_party = []
        
        for imp in imports:
            module = imp.get('module', imp.get('name', ''))
            base_module = module.split('.')[0] if module else ''
            
            if base_module in standard_libs:
                standard.append(imp)
            else:
                third_party.append(imp)
        
        return {
            'standard_library': standard,
            'third_party': third_party,
            'total': len(imports)
        }
    
    
    def _generate_quality_summary(
        self, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate code quality summary"""
        summary = analysis.get('summary', {})
        
        return {
            'quality_score': summary.get('quality_score', 0),
            'quality_grade': summary.get('quality_grade', 'N/A'),
            'pylint_score': analysis.get('pylint_score', 0),
            'total_issues': summary.get('total_issues', 0),
            'has_docstring': summary.get('has_docstring', False),
            'pylint_passed': summary.get('pylint_passed', False),
            'flake8_passed': summary.get('flake8_passed', False)
        }
    
    
    def _generate_function_signature(self, func: Dict[str, Any]) -> str:
        """Generate function signature string"""
        name = func['name']
        params = func.get('args', [])
        is_async = func.get('is_async', False)
        
        async_prefix = 'async ' if is_async else ''
        params_str = ', '.join(params) if params else ''
        
        return f"{async_prefix}def {name}({params_str})"
    
    
    def _format_as_markdown(self, doc: Dict[str, Any]) -> str:
        """Format documentation as Markdown"""
        md = []
        
        # Title
        md.append(f"# Documentation: {doc['filename']}\n")
        md.append(f"**Generated:** {doc['generated_at']}\n")
        md.append(f"**Language:** {doc['language']}\n")
        
        # Overview
        overview = doc['sections'].get('overview', {})
        md.append("\n## Overview\n")
        md.append(f"- **Total Lines:** {overview.get('total_lines', 0)}")
        md.append(f"- **Functions:** {overview.get('total_functions', 0)}")
        md.append(f"- **Classes:** {overview.get('total_classes', 0)}")
        md.append(f"- **Quality Score:** {overview.get('quality_score', 0)}/100 (Grade: {overview.get('quality_grade', 'N/A')})\n")
        
        # Description
        description = doc['sections'].get('description', '')
        if description:
            md.append("\n## Description\n")
            md.append(f"{description}\n")
        
        # Functions
        functions = doc['sections'].get('functions', [])
        if functions:
            md.append("\n## Functions\n")
            for func in functions:
                md.append(f"\n### `{func['signature']}`\n")
                md.append(f"**Lines:** {func['line_start']}-{func['line_end']}\n")
                
                if func.get('decorators'):
                    md.append(f"**Decorators:** {', '.join(func['decorators'])}\n")
                
                if func.get('parameters'):
                    md.append(f"**Parameters:** {', '.join(func['parameters'])}\n")
                
                md.append(f"\n**Docstring:**\n```markdown\n{func['docstring']}\n```\n")
        
        # Classes
        classes = doc['sections'].get('classes', [])
        if classes:
            md.append("\n## Classes\n")
            for cls in classes:
                md.append(f"\n### `class {cls['name']}`\n")
                md.append(f"**Lines:** {cls['line_start']}-{cls['line_end']}\n")
                
                if cls.get('base_classes'):
                    md.append(f"**Inherits from:** {', '.join(cls['base_classes'])}\n")
                
                md.append(f"**Methods:** {cls['method_count']}")
                if cls.get('methods'):
                    md.append(f" (`{', '.join(cls['methods'])}`)\n")
                else:
                    md.append("\n")
                
                md.append(f"\n**Docstring:**\n```markdown\n{cls.get('docstring', 'No docstring provided')}\n```\n")
        
        # Dependencies
        dependencies = doc['sections'].get('dependencies', {})
        if dependencies:
            md.append("\n## Dependencies\n")
            
            std_libs = dependencies.get('standard_library', [])
            if std_libs:
                md.append("\n### Standard Library\n")
                for imp in std_libs:
                    md.append(f"- `{imp.get('module', imp.get('name', 'unknown'))}`")
            
            third_party = dependencies.get('third_party', [])
            if third_party:
                md.append("\n### Third-Party\n")
                for imp in third_party:
                    md.append(f"- `{imp.get('module', imp.get('name', 'unknown'))}`")
        
        # Quality Summary
        quality = doc['sections'].get('quality_summary', {})
        md.append("\n## Code Quality Summary\n")
        md.append(f"- **Quality Score:** {quality.get('quality_score', 0)}/100")
        md.append(f"- **Quality Grade:** {quality.get('quality_grade', 'N/A')}")
        md.append(f"- **Pylint Score:** {quality.get('pylint_score', 0)}/10")
        md.append(f"- **Total Issues:** {quality.get('total_issues', 0)}")
        md.append(f"- **Has Module Docstring:** {'✅' if quality.get('has_docstring') else '❌'}")
        md.append(f"- **Pylint Passed:** {'✅' if quality.get('pylint_passed') else '❌'}")
        md.append(f"- **Flake8 Passed:** {'✅' if quality.get('flake8_passed') else '❌'}")
        
        return '\n'.join(md)
    
    
    def _format_as_html(self, doc: Dict[str, Any]) -> str:
        """Format documentation as HTML"""
        html = []
        
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("<meta charset='UTF-8'>")
        html.append(f"<title>Documentation: {doc['filename']}</title>")
        html.append("<style>")
        html.append(self._get_html_styles())
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Title
        html.append(f"<h1>Documentation: {doc['filename']}</h1>")
        html.append(f"<p><strong>Generated:</strong> {doc['generated_at']}</p>")
        html.append(f"<p><strong>Language:</strong> {doc['language']}</p>")
        
        # Overview
        overview = doc['sections'].get('overview', {})
        html.append("<h2>Overview</h2>")
        html.append("<ul>")
        html.append(f"<li><strong>Total Lines:</strong> {overview.get('total_lines', 0)}</li>")
        html.append(f"<li><strong>Functions:</strong> {overview.get('total_functions', 0)}</li>")
        html.append(f"<li><strong>Classes:</strong> {overview.get('total_classes', 0)}</li>")
        html.append(f"<li><strong>Quality Score:</strong> {overview.get('quality_score', 0)}/100 (Grade: {overview.get('quality_grade', 'N/A')})</li>")
        html.append("</ul>")
        
        # Description
        description = doc['sections'].get('description', '')
        if description:
            html.append("<h2>Description</h2>")
            # Note: For HTML output, conversion from Markdown (like **bold**) to HTML tags might be necessary here, 
            # but for simplicity, we treat the description as pre-formatted text block.
            html.append(f"<p>{description}</p>")
        
        # Functions
        functions = doc['sections'].get('functions', [])
        if functions:
            html.append("<h2>Functions</h2>")
            for func in functions:
                html.append(f"<h3><code>{func['signature']}</code></h3>")
                html.append(f"<p><strong>Lines:</strong> {func['line_start']}-{func['line_end']}</p>")
                html.append(f"<pre><code>{func['docstring']}</code></pre>")
        
        # Classes
        classes = doc['sections'].get('classes', [])
        if classes:
            html.append("<h2>Classes</h2>")
            for cls in classes:
                html.append(f"<h3><code>class {cls['name']}</code></h3>")
                html.append(f"<p><strong>Lines:</strong> {cls['line_start']}-{cls['line_end']}</p>")
                html.append(f"<p><strong>Methods:</strong> {cls['method_count']}</p>")
                html.append(f"<pre><code>{cls.get('docstring', 'No docstring provided')}</code></pre>")
        
        # Quality Summary
        quality = doc['sections'].get('quality_summary', {})
        html.append("<h2>Code Quality Summary</h2>")
        html.append("<ul>")
        html.append(f"<li><strong>Quality Score:</strong> {quality.get('quality_score', 0)}/100</li>")
        html.append(f"<li><strong>Pylint Score:</strong> {quality.get('pylint_score', 0)}/10</li>")
        html.append(f"<li><strong>Total Issues:</strong> {quality.get('total_issues', 0)}</li>")
        html.append("</ul>")
        
        html.append("</body>")
        html.append("</html>")
        
        return '\n'.join(html)
    
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML documentation"""
        return """
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
        pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
        pre code { background: transparent; padding: 0; }
        ul { list-style-type: none; padding-left: 0; }
        li { padding: 5px 0; }
        """


# ============================================================================
# Helper Functions
# ============================================================================

def generate_documentation(filepath: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to generate documentation
    """
    generator = DocumentationGenerator()
    return generator.generate_full_documentation(filepath, analysis_result)