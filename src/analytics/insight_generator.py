"""Automatic insight generator using Mistral Nemo 12B for fast analysis."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from typing import Optional


class AutoInsightGenerator:
    """Automatically generates performance insights using Mistral Nemo 12B."""
    
    def __init__(self, db_path: Path = Path("data/performance.db")):
        self.db_path = db_path
        self.last_analysis_count = 0
        self.min_queries_between_analysis = 10  # Analyze every 10 queries
        self.running = True
        
        # Background thread for periodic analysis
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
    
    def shutdown(self):
        """Graceful shutdown."""
        self.running = False
        if self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=10.0)
    
    def _analysis_loop(self):
        """Periodic analysis in background."""
        while self.running:
            try:
                # Run every 5 minutes
                time.sleep(300)
                self._generate_insights_if_needed()
            except Exception as e:
                print(f"[INSIGHT GEN ERROR] {e}")
    
    def _generate_insights_if_needed(self):
        """Check if new insights should be generated."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count total queries
                result = conn.execute("SELECT COUNT(*) FROM llm_performance").fetchone()
                total = result[0] if result else 0
                
                # Only generate if enough new data
                if total - self.last_analysis_count < self.min_queries_between_analysis:
                    return
                
                print(f"[INSIGHTS] Generating analysis for {total - self.last_analysis_count} new queries...")
                self.last_analysis_count = total
                
                # Generate insights
                self._generate_insights(conn)
        except Exception as e:
            print(f"[INSIGHT GEN ERROR] {e}")
    
    def _generate_insights(self, conn):
        """Generate automatic insights using Mistral Nemo 12B."""
        try:
            # Get recent performance data (last 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            
            # Summary stats by model and query type
            stats_query = """
                SELECT 
                    model,
                    query_type,
                    COUNT(*) as count,
                    AVG(tokens_per_second) as avg_tps,
                    AVG(completion_ms) as avg_ms,
                    AVG(completion_tokens) as avg_tokens
                FROM llm_performance
                WHERE timestamp > ? AND error IS NULL
                GROUP BY model, query_type
                ORDER BY count DESC
                LIMIT 20
            """
            
            results = conn.execute(stats_query, (cutoff,)).fetchall()
            
            if not results:
                print("[INSIGHTS] No data to analyze yet")
                return
            
            # Build data summary for LLM
            data_lines = []
            for row in results:
                model, qtype, count, avg_tps, avg_ms, avg_tokens = row
                qtype_str= qtype or "general"
                data_lines.append(
                    f"- {model} ({qtype_str}): {count} queries, "
                    f"{avg_tps:.1f} t/s, {avg_ms:.0f}ms avg, "
                    f"{avg_tokens:.0f} tokens/response"
                )
            
            data_summary = "\\n".join(data_lines)
            
            # Use Mistral Nemo 12B for FAST analysis (45 t/s vs 9 t/s for Qwen)
            from src.config.llm_config import load_llm_config
            from src.llm.client import LLMClient
            
            config = load_llm_config()
            # Override to use Qwen3-14B (fast model available locally)
            original_model = config.model_name
            config.model_name = "qwen3-14b"
            
            # Create client without analytics to avoid recursion
            client = LLMClient(config, enable_analytics=False)
            
            prompt = f"""Analyze this LLM performance data and provide 3-5 KEY insights:

{data_summary}

Focus on:
1. Best model for each query type (based on t/s and quality)
2. Performance patterns or anomalies
3. Optimization opportunities
4. Clear model selection recommendations

Be concise and actionable. Use bullet points."""

            messages = [
                {"role": "system", "content": "You are a performance analyst. Provide concise, actionable insights about LLM performance."},
                {"role": "user", "content": prompt}
            ]
            
            insights = client.generate_chat_completion(messages, temperature=0.3)
            
            # Store insights
            conn.execute("""
                INSERT INTO performance_insights (insight_type, content)
                VALUES (?, ?)
            """, ("auto_analysis", insights))
            
            conn.commit()
            
            print(f"[INSIGHTS] Generated new analysis using Mistral Nemo 12B")
            
        except Exception as e:
            print(f"[INSIGHT GENERATION ERROR] {e}")
