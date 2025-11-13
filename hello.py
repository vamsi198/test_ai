from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from bson import ObjectId
import traceback # Used for detailed error logging
import ast # Import for robust syntax validation
import re # Import regex for enhanced sanitization

# Import our custom modules
from config import Config
from models import get_db, CodeAnalysis, AnalysisHistory
from code_analyzer import analyze_code_file, calculate_metrics
from embeddings_service import EmbeddingService
from llm_service import LLMService 
from doc_generator import generate_documentation
from utils import allowed_file, cleanup_old_uploads, generate_unique_filename
from github_integration import GitHubIntegration, clone_and_analyze_repo

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize services
embedding_service = EmbeddingService()
llm_service = LLMService()
github_integration = GitHubIntegration()

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_DB_PATH'], exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS FOR LLM OUTPUT CLEANUP (FINAL ROBUST VERSION)
# ============================================================================

def sanitize_llm_code(text: str) -> str:
    """
    Cleans up LLM output for Python code.
    Only removes markdown fences and selectively strips leading error/explanation
    prefixes if they precede valid code. Preserves indentation.
    """
    # 1. Remove markdown code fences globally
    # Note: Using regex to capture '```' followed by optional language identifier (e.g., python)
    text = re.sub(r'```[a-z]*\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*$', '', text)
    
    lines = text.strip().splitlines()
    
    # Check if first line is an obvious error/note/explanation message
    if lines:
        first_line = lines[0].strip()
        
        if first_line.lower().startswith(('error:', 'note:', 'warning:', 'explanation:')):
            
            # Try to find where actual code starts (import, def, class, etc.)
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                if stripped and (
                    stripped.startswith(('import ', 'from ', 'def ', 'class ', '@')) or
                    (stripped.startswith('#') and not stripped.lower().startswith(('#error', '#note', '#warning', '#explanation'))) or
                    (i > 0 and lines[i-1].strip() == '') # Check if it's a new block starting after a blank line
                ):
                    # Found code start, return from this line onward, preserving indentation
                    return '\n'.join(lines[i:]).strip()
            
            # If we couldn't find a code start after an error prefix, just return the sanitized text
            # This handles cases where the entire output was just the error message.
            return text.strip()

    return text.strip()


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/api/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'AI Code Assistant',
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_code():
    """Upload code file for analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({
                'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        original_filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(original_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': unique_filename,
            'original_filename': original_filename,
            'filepath': filepath
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded code file"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        print(f"🔍 Analyzing file: {filename}")
        
        # Step 1: Static Analysis
        analysis_result = analyze_code_file(filepath)
        
        # Step 2: Calculate metrics
        metrics = calculate_metrics(filepath)
        
        # Step 3: Generate embeddings
        embeddings_result = embedding_service.process_code_file(filepath, filename)
        
        # Step 4: LLM review (with error handling)
        try:
            # Search for similar code for RAG context
            if embeddings_result['embeddings']:
                relevant_context = embedding_service.search_similar_code(
                    embeddings_result['embeddings'][0],
                    k=3
                )
            else:
                relevant_context = []
            
            llm_review = llm_service.generate_code_review(
                code_content=analysis_result['code'],
                static_analysis=analysis_result,
                relevant_context=relevant_context
            )
        except Exception as e:
            print(f"⚠️ LLM review failed: {e}")
            llm_review = {
                'review': 'LLM review unavailable',
                'suggestions': [],
                'security_concerns': []
            }

        # Step 4.5: LLM Code Update (IMPROVED VALIDATION + CONFIG SKIP + SANITIZATION)
        updated_code_text = ""
        original_code = analysis_result['code']
        refactoring_successful = False # <-- NEW FLAG
        
        # --- CONFIGURATION FILE CHECK ---
        skip_refactoring = any(keyword in filename.lower() for keyword in ['config', 'settings', '.env', 'secret', 'key'])
        
        if skip_refactoring:
            print(f"⚠️ Skipping code refactoring for {filename} (likely a configuration file)")
            updated_code_text = original_code # Show original code, no error message
        else:
            try:
                # IMPROVED PROMPT
                update_code_prompt = (
                    "You are a Python code refactoring assistant. "
                    "Rewrite the following Python code with improved readability, style, and best practices. "
                    "Return ONLY the complete, valid Python code. "
                    "Do NOT include explanations, markdown formatting, error messages, or any text other than code. "
                    "If you cannot refactor the code, return the code unchanged."
                )
                
                updated_code_text = llm_service.get_updated_code(
                    code_content=original_code,
                    prompt=update_code_prompt
                )
                
                # --- SANITIZATION STEP (CRITICAL FIX) ---
                updated_code_text = sanitize_llm_code(updated_code_text)
                
                # --------- ROBUST SYNTAX VALIDATION BLOCK (IMPROVED FALLBACK) ---------
                stripped = updated_code_text.strip()
                
                if not stripped:
                    print("⚠️ LLM returned empty output. Using original code.")
                    updated_code_text = original_code # <-- FALLBACK 1
                else:
                    try:
                        # Try to parse the entire output as valid Python code
                        ast.parse(stripped) 
                        
                        # If parsing succeeds:
                        print("✅ Updated code validated as proper Python syntax.")
                        refactoring_successful = True # <-- SET FLAG
                    except SyntaxError:
                        # If parsing fails:
                        print("⚠️ Updated code is not valid Python. Using original code as fallback.")
                        updated_code_text = original_code # <-- FALLBACK 2
                # ------------------------------------------------
                
            except Exception as e:
                print(f"❌ LLM update code failed: {e}")
                updated_code_text = original_code # <-- FALLBACK 3 (API failure)
                print("⚠️ Using original code due to API error.")
            
        # Step 5: Generate documentation
        documentation = generate_documentation(filepath, analysis_result)
        
        # Calculate summary scores
        total_issues = len(analysis_result.get('issues', []))
        pylint_score = analysis_result.get('pylint_score', 0)
        
        # Quality score calculation (0-100)
        quality_score = min(100, int(
            (pylint_score / 10 * 70) +  # 70% weight to pylint
            (max(0, 100 - total_issues * 5) * 0.3)  # 30% weight to issue count
        ))
        
        # Quality grade
        if quality_score >= 90:
            quality_grade = 'A'
        elif quality_score >= 80:
            quality_grade = 'B'
        elif quality_score >= 70:
            quality_grade = 'C'
        elif quality_score >= 60:
            quality_grade = 'D'
        else:
            quality_grade = 'F'
        
        # Build summary
        summary = {
            'quality_score': quality_score,
            'quality_grade': quality_grade,
            'pylint_score': pylint_score,
            'total_issues': total_issues,
            'total_functions': len(analysis_result.get('ast_analysis', {}).get('functions', [])),
            'total_classes': len(analysis_result.get('ast_analysis', {}).get('classes', [])),
            'total_lines': analysis_result.get('ast_analysis', {}).get('total_lines', 0),
            'has_docstring': bool(analysis_result.get('ast_analysis', {}).get('docstring')),
            'pylint_passed': pylint_score >= app.config['PYLINT_THRESHOLD'],
            'flake8_passed': analysis_result.get('flake8_passed', False)
        }
        
        # Combine all results
        complete_analysis = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'static_analysis': analysis_result,
            'metrics': metrics,
            'llm_review': llm_review,
            'documentation': documentation,
            'summary': summary,
            'pylint_score': pylint_score,
            # NEW FIELDS
            'updated_code': updated_code_text, 
            'refactoring_successful': refactoring_successful, # <-- ADDED FLAG
            'embedding_info': {
                'total_embeddings': len(embeddings_result['embeddings']),
                'functions_analyzed': len(embeddings_result['metadata'])
            }
        }
        
        # Save to MongoDB
        db = get_db()
        analysis_id = CodeAnalysis.create(db, complete_analysis)
        
        # Add to history
        AnalysisHistory.add_entry(db, {
            'analysis_id': str(analysis_id),
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'summary': summary
        })
        
        print(f"✅ Analysis complete. ID: {analysis_id}")
        
        # Cleanup old uploads
        cleanup_old_uploads(app.config['UPLOAD_FOLDER'], hours=24)
        
        return jsonify({
            'analysis_id': str(analysis_id),
            'message': 'Analysis completed successfully',
            'summary': summary
        }), 200
    
    except Exception as e:
        print(f"❌ Error in analyze: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<analysis_id>', methods=['GET'])
def get_results(analysis_id):
    """Get analysis results by ID. Now includes updated_code and refactoring_successful."""
    try:
        db = get_db()
        result = CodeAnalysis.get_by_id(db, analysis_id)
        
        if not result:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<analysis_id>', methods=['DELETE'])
def delete_result(analysis_id):
    """Delete an analysis by ID, the associated file, and the history record."""
    try:
        db = get_db()
        result = CodeAnalysis.get_by_id(db, analysis_id)
        if not result:
            return jsonify({'error': 'Analysis not found'}), 404

        # 1. Delete physical file
        filename = result.get("filename")
        if filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Warning: Could not delete file {filepath}: {e}")

        # 2. Delete main analysis record
        success = CodeAnalysis.delete(db, analysis_id)
        if not success:
            # Note: File deletion may have succeeded, but DB failed. Log warning.
            print(f"Warning: Failed to delete main analysis record for ID {analysis_id}")
            return jsonify({'error': 'DB delete failed (main analysis)'}), 500

        # 3. Delete history entry (FIX FOR HISTORY REAPPEARING)
        history_deleted = AnalysisHistory.delete_by_analysis_id(db, analysis_id)
        if not history_deleted:
            print(f"Warning: Could not delete history entry for analysis {analysis_id}")
            # Note: We still return success here, as the critical analysis record is gone.

        return jsonify({
            'success': True,
            'message': 'Analysis and file deleted successfully'
        }), 200

    except Exception as e:
        print(f"❌ Error in delete_result: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        
        db = get_db()
        history = AnalysisHistory.get_all(db, limit=limit, offset=offset)
        
        return jsonify({
            'history': history,
            'limit': limit,
            'offset': offset
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documentation/<analysis_id>', methods=['GET'])
def get_documentation(analysis_id):
    """Get generated documentation"""
    try:
        db = get_db()
        result = CodeAnalysis.get_by_id(db, analysis_id)
        
        if not result:
            return jsonify({'error': 'Analysis not found'}), 404
        
        documentation = result.get('documentation', {})
        
        return jsonify(documentation), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_similar_code():
    """Search for similar code using embeddings"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
        k = data.get('k', 5)
        
        # Generate embedding for query text
        # Note: If embedding service uses an LLM for query, this would be different
        # Assuming model.encode is the correct method for the current EmbeddingService setup
        query_embedding = embedding_service.model.encode([query])[0] 
        
        # Search
        results = embedding_service.search_similar_code(query_embedding, k=k)
        
        return jsonify({
            'query': query,
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    try:
        db = get_db()
        stats = CodeAnalysis.get_statistics(db)
        
        # Add embedding count
        stats['total_embeddings'] = embedding_service.get_total_embeddings()
        
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# GITHUB INTEGRATION ROUTES
# ============================================================================

@app.route('/api/github/clone', methods=['POST'])
def clone_github_repository():
    """
    Clone and analyze a GitHub repository
    Expected JSON: { "repo_url": "[https://github.com/user/repo](https://github.com/user/repo)" }
    """
    try:
        data = request.get_json()
        repo_url = data.get('repo_url')

        if not repo_url:
            return jsonify({'error': 'Repository URL required'}), 400

        if not app.config['ENABLE_GIT_INTEGRATION']:
            return jsonify({'error': 'Git integration is disabled. Set ENABLE_GIT_INTEGRATION=True in .env'}), 403

        print(f"🔄 Cloning repository: {repo_url}")

        # Clone repository
        clone_result = github_integration.clone_repository(repo_url)

        if not clone_result.get('success'):
            return jsonify(clone_result), 400

        return jsonify({
            'success': True,
            'message': 'Repository cloned successfully',
            'repo_url': repo_url,
            'total_files': clone_result.get('total_files', 0),
            'files_found': clone_result.get('total_files', 0),
            'clone_path': clone_result.get('clone_path', ''),
            'branch': clone_result.get('branch', 'main'),
            'files': [f.replace('\\', '/').split('/')[-1] for f in clone_result.get('files', [])[:10]] 
        }), 200

    except Exception as e:
        print(f"❌ Error cloning repo: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/github/pr/comment', methods=['POST'])
def post_pr_comment():
    """
    Post analysis comment on GitHub PR
    Expected JSON: {
        "repo_owner": "username",
        "repo_name": "repository",
        "pr_number": 123,
        "analysis_id": "mongo_id"
    }
    """
    try:
        data = request.get_json()

        repo_owner = data.get('repo_owner')
        repo_name = data.get('repo_name')
        pr_number = data.get('pr_number')
        analysis_id = data.get('analysis_id')

        if not all([repo_owner, repo_name, pr_number, analysis_id]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Get analysis results
        db = get_db()
        analysis = CodeAnalysis.get_by_id(db, analysis_id)

        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404

        # Format as comment
        comment_body = github_integration.format_analysis_as_comment(analysis)

        # Post comment
        result = github_integration.post_pr_comment(
            repo_owner=repo_owner,
            repo_name=repo_name,
            pr_number=pr_number,
            comment_body=comment_body,
            platform='github'
        )

        return jsonify(result), 200 if result.get('success') else 400

    except Exception as e:
        print(f"❌ Error posting PR comment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/github/pr/files', methods=['GET'])
def get_pr_files():
    """
    Get files changed in a PR
    Query params: ?owner=username&repo=reponame&pr=123
    """
    try:
        repo_owner = request.args.get('owner')
        repo_name = request.args.get('repo')
        pr_number = request.args.get('pr')

        if not all([repo_owner, repo_name, pr_number]):
            return jsonify({'error': 'Missing required parameters'}), 400

        result = github_integration.get_pr_files(
            repo_owner=repo_owner,
            repo_name=repo_name,
            pr_number=int(pr_number),
            platform='github'
        )

        return jsonify(result), 200 if result.get('success') else 400

    except Exception as e:
        print(f"❌ Error getting PR files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/github/webhook/setup', methods=['POST'])
def setup_github_webhook():
    """
    Setup GitHub webhook for automated PR analysis
    Expected JSON: {
        "repo_owner": "username",
        "repo_name": "repository",
        "webhook_url": "[https://your-server.com/api/github/webhook](https://your-server.com/api/github/webhook)",
        "events": ["pull_request", "push"]
    }
    """
    try:
        data = request.get_json()

        repo_owner = data.get('repo_owner')
        repo_name = data.get('repo_name')
        webhook_url = data.get('webhook_url')
        events = data.get('events', ['pull_request'])

        if not all([repo_owner, repo_name, webhook_url]):
            return jsonify({'error': 'Missing required fields'}), 400

        result = github_integration.setup_webhook(
            repo_owner=repo_owner,
            repo_name=repo_name,
            webhook_url=webhook_url,
            events=events,
            platform='github'
        )

        return jsonify(result), 200 if result.get('success') else 400

    except Exception as e:
        print(f"❌ Error setting up webhook: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/github/webhook', methods=['POST'])
def github_webhook_handler():
    """
    Handle GitHub webhook events
    Automatically triggered by GitHub when PR is opened/updated
    """
    try:
        # Get GitHub event type
        event_type = request.headers.get('X-GitHub-Event')

        if event_type != 'pull_request':
            return jsonify({'message': 'Event ignored'}), 200

        data = request.get_json()

        # Extract PR info
        pr = data.get('pull_request', {})
        action = data.get('action')

        # Only analyze on opened or synchronize (new commits)
        if action not in ['opened', 'synchronize']:
            return jsonify({'message': 'Action ignored'}), 200

        repo_owner = data['repository']['owner']['login']
        repo_name = data['repository']['name']
        pr_number = pr['number']
        
        print(f"📥 Webhook: PR #{pr_number} in {repo_owner}/{repo_name} (Action: {action})")

        # Get changed files
        files_result = github_integration.get_pr_files(
            repo_owner=repo_owner,
            repo_name=repo_name,
            pr_number=pr_number,
            platform='github'
        )

        if not files_result.get('success'):
            return jsonify(files_result), 400

        # Filter Python files
        python_files = [
            f for f in files_result.get('files', [])
            if f['filename'].endswith('.py')
        ]

        if not python_files:
            # Post a comment saying no Python files found
            comment = f"## 🤖 AI Code Review\n\nNo Python files found in this PR."
            github_integration.post_pr_comment(
                repo_owner=repo_owner,
                repo_name=repo_name,
                pr_number=pr_number,
                comment_body=comment,
                platform='github'
            )
            return jsonify({'message': 'No Python files in PR'}), 200

        print(f"📄 Found {len(python_files)} Python files to analyze")

        # Simple acknowledgment (real implementation would be async)
        # For production, use Celery/background tasks
        
        # Post initial comment
        initial_comment = f"""## 🤖 AI Code Review In Progress
Analyzing {len(python_files)} Python file(s):
{chr(10).join([f'- `{f["filename"]}`' for f in python_files[:5]])}
Analysis will be posted shortly..."""
        github_integration.post_pr_comment(
            repo_owner=repo_owner,
            repo_name=repo_name,
            pr_number=pr_number,
            comment_body=initial_comment,
            platform='github'
        )

        return jsonify({
            'message': 'Webhook processed',
            'pr_number': pr_number,
            'python_files_found': len(python_files),
            'status': 'Analysis queued'
        }), 200

    except Exception as e:
        print(f"❌ Webhook error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("🚀 Starting AI Code Assistant Backend...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🗄️ Vector DB path: {app.config['VECTOR_DB_PATH']}")
    print(f"🌐 Server running on http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )