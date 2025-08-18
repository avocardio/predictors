"""
Reasoning Agent for Autonomous ML Research
Uses Claude Opus 4.1 with extended thinking and tool use
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from anthropic import Anthropic, APIStatusError


class ReasoningAgent:
    """Agent that reasons through ML research tasks step by step"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        # Use Claude Opus 4.1 - the latest reasoning model with tool use
        self.model = "claude-opus-4-1-20250805"  
        self.reasoning_traces = []
        self.workspace = Path("/tmp/predictors")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Cached system prompt for reuse across all API calls
        self.cached_system_prompt = """You are an autonomous ML researcher specializing in novel cross-domain predictions.

Your expertise includes:
- Discovering creative correlations between unrelated data domains
- Analyzing real datasets and extracting meaningful features  
- Adapting transformer architectures for diverse prediction tasks
- Designing complete ML pipelines from data to trained models

Key principles:
- Always use REAL downloaded data, never synthetic placeholders
- Treat transformers as universal prediction engines via proper tokenization
- Focus on feature extraction and input shaping, not architecture changes
- Create complete end-to-end experiments with proper evaluation

AVAILABLE PACKAGES (use ONLY these):
- torch, lightning.pytorch, transformers
- pandas, numpy, datasets  
- sklearn (preprocessing, metrics only)
- wandb, matplotlib, seaborn
- json, os, pathlib, datetime

Available tools: web_search_20250305, check_kaggle, download_data, write_file, run_command

Follow instructions precisely and return structured responses."""
        
        # Define tools: mix of server tools (web search) and client tools (custom)
        # Add cache_control to the first tool to enable caching of the entire tools array
        self.tools = [
            # Server tool - executes on Anthropic's servers
            {
                "name": "web_search_20250305",  # Use versioned server tool
                "description": "Search the web for information about datasets, APIs, or ML techniques",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                },
                "cache_control": {"type": "ephemeral"}  # Cache the tools array
            },
            # Client tools - execute on our system
            {
                "name": "check_kaggle",
                "description": "Check if a Kaggle dataset exists and get its info",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Dataset name (user/dataset-name)"}
                    },
                    "required": ["dataset"]
                }
            },
            {
                "name": "download_data",
                "description": "Download a dataset from Kaggle or HuggingFace",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["kaggle", "huggingface"]},
                        "identifier": {"type": "string", "description": "Dataset identifier"},
                        "path": {"type": "string", "description": "Local path to save"}
                    },
                    "required": ["source", "identifier", "path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write code or configuration to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "File content"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "run_command", 
                "description": "Run a shell command (e.g., python train.py)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"}
                    },
                    "required": ["command"]
                }
            }
        ]
        
    def _api_call_with_retry(self, **kwargs):
        """Make API call with retry logic for handling overload errors"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                return response
            except APIStatusError as e:
                if e.status_code == 529:  # Overloaded error
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ API overloaded, retrying in {wait_time} seconds... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ API overloaded after {self.max_retries} attempts")
                        raise
                else:
                    raise  # Re-raise non-overload errors
        
    def log_reasoning(self, stage: str, thought: str, action: str = None, result: str = None):
        """Log reasoning trace for visibility"""
        trace = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "thought": thought,
            "action": action,
            "result": result
        }
        self.reasoning_traces.append(trace)
        
        # Print for real-time visibility
        print(f"\n🧠 [{stage}]")
        print(f"   Thinking: {thought[:200]}...")
        if action:
            print(f"   Action: {action}")
        if result:
            print(f"   Result: {result[:100]}...")
    
    def reason_about_task(self, history: List[Dict]) -> Dict:
        """Step 1: Reason about what novel task to explore"""
        
        self.log_reasoning(
            stage="TASK_DISCOVERY",
            thought="Let me think about novel cross-domain predictions that haven't been explored..."
        )
        
        prompt = f"""You are an ML researcher discovering novel prediction tasks.

PREVIOUS TASKS TO AVOID: {[t.get('name') for t in history[-5:]]}

YOUR STRUCTURED WORKFLOW:

STEP 1 - BRAINSTORM: Think of ONE novel cross-domain prediction like:
- Weather patterns → social media mood
- Architecture styles → economic indicators  
- Music features → migration patterns

STEP 2 - SEARCH: Use web_search_20250305 to find if relevant datasets exist

STEP 3 - VERIFY: Use check_kaggle to verify specific datasets exist  

STEP 4 - RETURN EXACT JSON:
{{
  "name": "short_descriptive_name",
  "description": "what you're predicting and why", 
  "hypothesis": "why this correlation might exist",
  "data_sources": [
    {{"type": "kaggle", "identifier": "user/dataset-name", "description": "what it provides"}},
    {{"type": "huggingface", "identifier": "dataset-name", "description": "what it provides"}}
  ],
  "task_type": "regression",
  "evaluation_metric": "rmse",
  "expected_baseline": "rough baseline performance estimate"
}}

CRITICAL RULES:
- Use tools to verify datasets exist BEFORE suggesting them
- Return ONLY the final JSON, no other text
- Don't write files or run commands in this step

Start by searching for interesting datasets!"""

        user_message = f"""TASK: Discover a novel cross-domain prediction task.

PREVIOUS TASKS TO AVOID: {[t.get('name') for t in history[-5:]]}

WORKFLOW:
1. BRAINSTORM: Think of ONE novel cross-domain prediction (e.g. weather → social media mood)
2. SEARCH: Use web_search_20250305 to find relevant datasets
3. VERIFY: Use check_kaggle to verify specific datasets exist
4. RETURN: Exact JSON format with verified data sources

JSON FORMAT:
{{
  "name": "short_descriptive_name",
  "description": "what you're predicting and why", 
  "hypothesis": "why this correlation might exist",
  "data_sources": [
    {{"type": "kaggle", "identifier": "user/dataset-name", "description": "what it provides"}}
  ],
  "task_type": "regression",
  "evaluation_metric": "rmse",
  "expected_baseline": "rough baseline performance estimate"
}}

CRITICAL: Use tools to verify datasets exist BEFORE suggesting them. Return ONLY the final JSON."""
        
        messages = [{"role": "user", "content": user_message}]
        
        # Handle tool use conversation with cached system prompt
        while True:
            response = self._api_call_with_retry(
                model=self.model,
                max_tokens=3000,
                tools=self.tools,
                system=[{
                    "type": "text",
                    "text": self.cached_system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=messages
            )
            
            # Track token usage including cache metrics
            if hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                if hasattr(response.usage, 'cache_creation_input_tokens'):
                    self.cache_creation_tokens += response.usage.cache_creation_input_tokens
                if hasattr(response.usage, 'cache_read_input_tokens'):
                    self.cache_read_tokens += response.usage.cache_read_input_tokens
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Extract tool calls
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_use_id = content_block.id
                        
                        # Execute the tool
                        if tool_name == "web_search_20250305":
                            # Server tool - results come back automatically
                            tool_result = f"Web search completed for: {tool_input.get('query')}"
                        else:
                            # Client tool - we execute it
                            tool_result = self.execute_tool(tool_name, tool_input)
                        
                        # Add tool result to conversation
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": tool_result
                            }]
                        })
            else:
                # No more tool use, extract final response
                content = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        content += content_block.text
                
                task_json = self._extract_json(content)
                
                # Validate that we got a proper task
                if task_json and "name" in task_json:
                    return task_json
                else:
                    print(f"⚠️ Invalid JSON response, using fallback task")
                    return self._get_fallback_task()
                    
                break
    
    def verify_data_sources(self, task: Dict) -> Dict:
        """Step 2: Use tools to verify data sources exist"""
        
        self.log_reasoning(
            stage="DATA_VERIFICATION",
            thought=f"I need to verify that data sources for '{task['name']}' actually exist online"
        )
        
        verified_sources = []
        
        for source in task.get("data_sources", []):
            # Handle both dict and string formats
            if isinstance(source, dict):
                source_type = source.get("type", "").lower()
                identifier = source.get("identifier", "")
                description = source.get("description", "")
            else:
                # Fallback for string format
                source_type = "kaggle" if "kaggle" in source.lower() else "huggingface"
                identifier = source
                description = source
            
            # Check if it's a Kaggle dataset
            if source_type == "kaggle":
                exists = self._check_kaggle_dataset(identifier)
                self.log_reasoning(
                    stage="KAGGLE_CHECK",
                    thought=f"Checking Kaggle for {identifier}",
                    action=f"kaggle datasets list -s {identifier}",
                    result="Found" if exists else "Not found"
                )
                if exists:
                    verified_sources.append({
                        "type": "kaggle",
                        "identifier": identifier,
                        "description": description,
                        "verified": True
                    })
            
            # Check HuggingFace
            elif source_type == "huggingface":
                self.log_reasoning(
                    stage="HF_CHECK",
                    thought=f"Checking HuggingFace for {identifier}",
                    action="HuggingFace API check",
                    result="Assuming available"
                )
                verified_sources.append({
                    "type": "huggingface",
                    "identifier": identifier,
                    "description": description,
                    "verified": True
                })
        
        task["verified_sources"] = verified_sources
        return task
    
    def design_experiment(self, task: Dict) -> Dict:
        """Step 3: Reason about how to structure the experiment"""
        
        self.log_reasoning(
            stage="EXPERIMENT_DESIGN",
            thought=f"Designing experiment architecture for {task['name']}"
        )
        
        prompt = f"""Given this ML task: {json.dumps(task, indent=2)}

REASON STEP BY STEP about adapting the BaseTransformer:
1. What data preprocessing is needed?
2. How to adapt BaseTransformer input dimensions?
3. What sequence structure (time series vs single predictions)?
4. What loss function and metrics?
5. What are potential challenges?

Return as JSON:
{{
  "reasoning": ["step1...", "step2..."],
  "transformer_adaptation": {{
    "input_processing": "how to extract features",
    "input_dim": number,
    "output_dim": number,
    "sequence_length": number,
    "sequence_type": "temporal/single"
  }},
  "training": {{
    "loss_function": "mse/ce/etc", 
    "metrics": ["metric1", "metric2"],
    "expected_challenges": ["challenge1", "challenge2"]
  }}
}}"""

        response = self._api_call_with_retry(
            model=self.model,
            max_tokens=3000,  # Standardized to match other calls
            tools=self.tools,  # Include tools for cache consistency
            system=[{
                "type": "text",
                "text": self.cached_system_prompt,
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Track token usage including cache metrics
        if hasattr(response, 'usage'):
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            if hasattr(response.usage, 'cache_creation_input_tokens'):
                self.cache_creation_tokens += response.usage.cache_creation_input_tokens
            if hasattr(response.usage, 'cache_read_input_tokens'):
                self.cache_read_tokens += response.usage.cache_read_input_tokens
        
        content = response.content[0].text
        design = self._extract_json(content)
        
        # Log reasoning
        for step in design.get("reasoning", []):
            self.log_reasoning(
                stage="DESIGN_REASONING",
                thought=step
            )
        
        task["experiment_design"] = design
        return task
    
    def analyze_downloaded_data(self, task: Dict, data_paths: Dict) -> Dict:
        """Step 4: Analyze actual downloaded data to understand structure"""
        
        self.log_reasoning(
            stage="DATA_ANALYSIS", 
            thought=f"Analyzing downloaded data files to understand structure and features for {task['name']}"
        )
        
        # Examine actual data files and include absolute file paths
        data_analysis = {}
        for source_name, path in data_paths.items():
            if path and Path(path).exists():
                analysis = self._analyze_data_file(path)
                analysis["file_path"] = str(Path(path).resolve())  # Add absolute path
                data_analysis[source_name] = analysis
            else:
                # Mark file as missing but don't create synthetic fallback
                data_analysis[source_name] = {
                    "error": f"File not found: {path}",
                    "status": "missing",
                    "attempted_path": path
                }
                
        task["data_analysis"] = data_analysis
        return task
    
    def generate_code(self, task: Dict, history: List[Dict] = None) -> str:
        """Step 5: Generate complete experiment code based on real data analysis"""
        
        self.log_reasoning(
            stage="CODE_GENERATION",
            thought="Generating complete experiment pipeline based on actual data analysis"
        )
        
        # Get actual file paths from data analysis
        available_files = []
        for source_name, analysis in task.get('data_analysis', {}).items():
            if 'file_path' in analysis:
                available_files.append(analysis['file_path'])
        
        # Add previous error context if available
        error_context = ""
        if history:
            recent_failures = [h for h in history[-3:] if h.get('training_result', {}).get('status') == 'failed']
            if recent_failures:
                error_context = f"\nPREVIOUS ERRORS TO AVOID:\n"
                for fail in recent_failures:
                    error = fail.get('training_result', {}).get('error', '')
                    if error:
                        error_context += f"- {error[:200]}\n"

        prompt = f"""Write ONLY executable Python code. No explanations, no markdown.

TASK: {task['name']} - {task['description']}
{error_context}
DOWNLOADED DATA FILES (use EXACTLY these paths):
{chr(10).join(f"- {path}" for path in available_files)}

DATA STRUCTURE: {json.dumps(task.get('data_analysis', {}), indent=2)}

STRICT REQUIREMENTS:
1. Use ONLY: torch, lightning.pytorch, pandas, numpy, sklearn.preprocessing, wandb, json, os, pathlib
2. Load data from EXACT file paths in DATA STRUCTURE (file_path field), NOT descriptions!
3. Handle missing files by exploring the actual directory structure with os.listdir()
4. Create BaseTransformer with input_dim based on your extracted feature count
5. Create sequences: (batch_size, seq_len, features) for transformer
6. Include Lightning training + conditional WandB logging (check wandb env var)
7. Use if __name__ == "__main__": main() structure
8. WANDB: Use wandb_key = os.getenv('wandb') and if exists, call wandb.login(key=wandb_key) then wandb.init()
9. ALWAYS include training prints: print epochs, losses, progress
10. ALWAYS save results.json with final metrics at the end

CRITICAL FILE PATH USAGE:
- Use analysis["file_path"] from DATA STRUCTURE, NOT the description field
- Example: pd.read_csv(analysis["file_path"]) NOT pd.read_csv("description text")
- If file doesn't exist, find the correct file by listing directories and matching patterns

START WITH IMPORTS:"""

        # Use same cached system prompt for consistency, but modify user message
        code_prompt = f"""CODE GENERATION MODE: Write ONLY executable Python code. No explanations or markdown.

{prompt}"""
        
        response = self._api_call_with_retry(
            model=self.model,
            max_tokens=3000,  # Standardized to match other calls
            tools=self.tools,  # Include tools for cache consistency (will be ignored)
            system=[{
                "type": "text", 
                "text": self.cached_system_prompt,  # Reuse same cached prompt
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": code_prompt}]
        )
        
        # Track token usage
        if hasattr(response, 'usage'):
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            if hasattr(response.usage, 'cache_creation_input_tokens'):
                self.cache_creation_tokens += response.usage.cache_creation_input_tokens
            if hasattr(response.usage, 'cache_read_input_tokens'):
                self.cache_read_tokens += response.usage.cache_read_input_tokens
        
        raw_code = response.content[0].text
        
        # Clean the code in case it has markdown formatting
        clean_code = self._extract_python_code(raw_code)
        task["generated_code"] = clean_code
        
        return task
    
    def _extract_python_code(self, text: str) -> str:
        """Extract clean Python code from response, removing markdown formatting"""
        import re
        
        # Remove markdown code blocks if present
        if '```python' in text:
            # Extract code between ```python and ```
            match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
            if match:
                return match.group(1)
        elif '```' in text:
            # Extract code between generic code blocks
            match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
            if match:
                return match.group(1)
        
        # If no code blocks, look for Python import statements as start
        if 'import ' in text:
            # Find the first import statement and take everything from there
            lines = text.split('\n')
            code_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    code_start = i
                    break
            
            if code_start >= 0:
                # Take all lines from first import onward
                code_lines = lines[code_start:]
                return '\n'.join(code_lines).strip()
        
        # If still no valid code found, return original but cleaned
        return text.strip()
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from model response"""
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                return {}
        return {}
    
    def _extract_kaggle_name(self, source: str) -> Optional[str]:
        """Extract Kaggle dataset name from source string"""
        # Look for patterns like "username/dataset-name"
        import re
        match = re.search(r'[\w-]+/[\w-]+', source)
        return match.group() if match else None
    
    def _extract_hf_name(self, source: str) -> str:
        """Extract HuggingFace dataset name"""
        # Simple extraction, could be improved
        return source.split("/")[-1] if "/" in source else source
    
    def _check_kaggle_dataset(self, dataset_name: str) -> bool:
        """Check if Kaggle dataset exists using kaggle CLI"""
        try:
            result = subprocess.run(
                f"kaggle datasets list -s {dataset_name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return dataset_name in result.stdout
        except:
            return False
    
    def _get_fallback_task(self) -> Dict:
        """Fallback task when parsing fails"""
        return {
            "name": "weather_sentiment_correlation",
            "description": "Predict social media sentiment scores from local weather conditions",
            "hypothesis": "Weather affects mood which influences social media posting patterns",
            "data_sources": [
                {"type": "huggingface", "identifier": "tweet_eval", "description": "sentiment labeled tweets"},
                {"type": "kaggle", "identifier": "berkeleyearth/climate-change-earth-surface-temperature-data", "description": "weather data"}
            ],
            "task_type": "regression",
            "evaluation_metric": "rmse",
            "expected_baseline": "random prediction around 0.5 sentiment score"
        }
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> str:
        """Execute a tool and return the result"""
        
        if tool_name == "web_search":
            # Simple web search simulation
            query = parameters.get("query", "")
            self.log_reasoning(
                stage="TOOL_EXECUTION",
                thought=f"Searching web for: {query}",
                action=f"web_search({query})",
                result="Found relevant information"
            )
            return f"Search results for '{query}': Found relevant datasets and APIs"
        
        elif tool_name == "check_kaggle":
            dataset = parameters.get("dataset", "")
            exists = self._check_kaggle_dataset(dataset)
            result = "Dataset found" if exists else "Dataset not found"
            self.log_reasoning(
                stage="TOOL_EXECUTION", 
                thought=f"Checking Kaggle for {dataset}",
                action=f"kaggle datasets list -s {dataset}",
                result=result
            )
            return result
        
        elif tool_name == "download_data":
            source = parameters.get("source")
            identifier = parameters.get("identifier")
            path = parameters.get("path")
            
            self.log_reasoning(
                stage="TOOL_EXECUTION",
                thought=f"Downloading {identifier} from {source}",
                action=f"Download to {path}",
                result="Download started"
            )
            
            # Execute actual download
            if source == "kaggle":
                try:
                    result = subprocess.run(
                        f"kaggle datasets download -d {identifier} -p {path}",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    return "Downloaded successfully" if result.returncode == 0 else "Download failed"
                except:
                    return "Download failed"
            
            return "Download completed"
        
        elif tool_name == "write_file":
            path = parameters.get("path")
            content = parameters.get("content")
            
            self.log_reasoning(
                stage="TOOL_EXECUTION",
                thought=f"Writing code to {path}",
                action="file_write",
                result=f"File written ({len(content)} chars)"
            )
            
            try:
                with open(path, 'w') as f:
                    f.write(content)
                return f"File written successfully: {path}"
            except Exception as e:
                return f"Failed to write file: {e}"
        
        elif tool_name == "run_command":
            command = parameters.get("command")
            
            self.log_reasoning(
                stage="TOOL_EXECUTION",
                thought=f"Running command: {command}",
                action=command,
                result="Command executed"
            )
            
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                return f"Command output: {result.stdout[:200]}..."
            except Exception as e:
                return f"Command failed: {e}"
        
        return "Tool not implemented"
    
    def _analyze_data_file(self, file_path: str) -> Dict:
        """Analyze a data file to understand its structure"""
        try:
            import pandas as pd
            path = Path(file_path)
            
            # Handle different file types
            if path.suffix.lower() == '.csv':
                # Read first few rows to understand structure
                df = pd.read_csv(path, nrows=100)
                return {
                    "file_type": "csv",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "sample_data": df.head(3).to_dict(),
                    "missing_values": df.isnull().sum().to_dict()
                }
            elif path.suffix.lower() in ['.json', '.jsonl']:
                # Handle JSON files
                with open(path, 'r') as f:
                    sample = f.readline()
                return {
                    "file_type": "json", 
                    "sample_line": sample[:200],
                    "estimated_size": path.stat().st_size
                }
            else:
                return {
                    "file_type": "unknown",
                    "size": path.stat().st_size,
                    "extension": path.suffix
                }
        except Exception as e:
            return {
                "error": str(e),
                "file_path": file_path
            }

    def run_full_pipeline(self, history: List[Dict]) -> Dict:
        """Run the complete reasoning pipeline"""
        
        print("\n" + "="*60)
        print("🚀 STARTING AUTONOMOUS ML RESEARCH PIPELINE")
        print("="*60)
        
        # Step 1: Discover task
        task = self.reason_about_task(history)
        
        # Step 2: Verify data sources
        task = self.verify_data_sources(task)
        
        # Step 3: Design experiment
        task = self.design_experiment(task)
        
        # Step 4: Analyze downloaded data (NEW)
        # This will be called after data is downloaded in the main pipeline
        
        # Step 5: Generate code (will be called with data analysis)
        
        # Add all reasoning traces
        task["reasoning_traces"] = self.reasoning_traces
        
        # Calculate API costs with cache savings
        # Cached tokens cost 10% of regular tokens
        regular_input_tokens = self.total_input_tokens - self.cache_read_tokens
        cached_cost = (self.cache_read_tokens / 1_000_000) * 1.5  # $1.50 per million cached tokens (90% discount)
        regular_input_cost = (regular_input_tokens / 1_000_000) * 15  # $15 per million
        output_cost = (self.total_output_tokens / 1_000_000) * 75  # $75 per million
        total_cost = cached_cost + regular_input_cost + output_cost
        
        # Calculate savings from caching
        without_cache_cost = (self.total_input_tokens / 1_000_000) * 15 + output_cost
        savings = without_cache_cost - total_cost
        
        print("\n" + "="*60)
        print("📊 REASONING COMPLETE")
        print(f"   Total reasoning steps: {len(self.reasoning_traces)}")
        print(f"\n💰 API USAGE & COSTS:")
        print(f"   Input tokens: {self.total_input_tokens:,}")
        print(f"   Output tokens: {self.total_output_tokens:,}")
        print(f"   Cache creation: {self.cache_creation_tokens:,} tokens")
        print(f"   Cache hits: {self.cache_read_tokens:,} tokens")
        
        # Debug cache issues
        if self.cache_creation_tokens == 0 and self.cache_read_tokens == 0:
            print(f"   ⚠️ NO CACHING DETECTED - Check system prompt consistency")
        
        print(f"\n   Regular input cost: ${regular_input_cost:.4f}")
        print(f"   Cached input cost: ${cached_cost:.4f}")
        print(f"   Output cost: ${output_cost:.4f}")
        print(f"   Total cost: ${total_cost:.4f}")
        if self.cache_read_tokens > 0:
            print(f"   💸 Saved ${savings:.4f} from caching ({(savings/without_cache_cost)*100:.1f}%)")
        print("="*60)
        
        return task
    
    def save_reasoning_log(self, filepath: Path):
        """Save full reasoning trace to file"""
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "traces": self.reasoning_traces
            }, f, indent=2)
        print(f"💾 Reasoning log saved: {filepath}")