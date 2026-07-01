#!/usr/bin/env python3
import os
import sys
import argparse
import json
import yaml
from pathlib import Path

# Add project root to path to ensure src can be imported if needed
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def load_config():
    """Load config from config/default.yaml or return default settings."""
    config_path = project_root / "config" / "default.yaml"
    defaults = {
        "persist_dir": "data/memory",
        "collection_name": "ai_actor_memories"
    }
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                memory_config = config.get("memory", {})
                defaults["persist_dir"] = memory_config.get("persist_directory", "data/memory")
                defaults["collection_name"] = memory_config.get("collection_name", "ai_actor_memories")
        except Exception as e:
            print(f"Warning: Failed to load config/default.yaml: {e}. Using defaults.")
    
    # Resolve relative persist path to project root
    if not os.path.isabs(defaults["persist_dir"]):
        defaults["persist_dir"] = str(project_root / defaults["persist_dir"])
        
    return defaults

# Load the config options
db_config = load_config()

def get_chroma_client_and_collection(persist_dir, collection_name):
    """Initialize persistent Chroma client and return client, collection."""
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

# CLI Mode implementation
def run_cli(persist_dir, collection_name):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    import chromadb

    console = Console()
    console.print(Panel.fit("[bold cyan]ChromaDB Memory Explorer (CLI Mode)[/bold cyan]", border_style="cyan"))
    console.print(f"📁 [bold]Persist Directory:[/bold] {persist_dir}")
    console.print(f"📦 [bold]Collection Name:[/bold] {collection_name}")

    try:
        client, collection = get_chroma_client_and_collection(persist_dir, collection_name)
        count = collection.count()
        console.print(f"✨ [bold green]Connected successfully![/bold green] Total documents in collection: [bold yellow]{count}[/bold yellow]\n")
        
        if count == 0:
            console.print("[yellow]The collection is empty.[/yellow]")
            return

        # Fetch records
        results = collection.get()
        table = Table(title="Memories stored in ChromaDB", show_lines=True)
        table.add_column("Index", justify="right", style="cyan", no_wrap=True)
        table.add_column("ID", style="blue")
        table.add_column("Document Preview", style="green")
        table.add_column("Age Stage", style="magenta")
        table.add_column("Type", style="yellow")
        table.add_column("Emotion", style="red")
        table.add_column("Importance", style="white")

        for idx in range(len(results["ids"])):
            mem_id = results["ids"][idx]
            doc = results["documents"][idx] if results["documents"] else ""
            meta = results["metadatas"][idx] if results["metadatas"] else {}
            
            # Truncate document preview
            doc_preview = doc[:80] + "..." if len(doc) > 80 else doc
            
            table.add_row(
                str(idx + 1),
                mem_id,
                doc_preview,
                str(meta.get("age_stage", "N/A")),
                str(meta.get("memory_type", "N/A")),
                str(meta.get("emotional_tag", "N/A")),
                f"{meta.get('importance', 0.5):.2f}"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error accessing ChromaDB: {e}[/bold red]")
        sys.exit(1)

# Web UI Mode implementation using Gradio
def run_web_ui(persist_dir, collection_name, port=7860):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required for the Web UI. Install it with: pip install gradio")
        sys.exit(1)

    print(f"Starting Gradio Web UI server. Database: {persist_dir}, Collection: {collection_name}")

    # Helper to load data
    def load_data(filter_age=None, filter_type=None):
        try:
            _, collection = get_chroma_client_and_collection(persist_dir, collection_name)
            count = collection.count()
            if count == 0:
                return [], f"Database is empty. Persistent Directory: {persist_dir}"

            # Retrieve all records
            all_records = collection.get()
            
            # Format rows
            rows = []
            for idx in range(len(all_records["ids"])):
                mem_id = all_records["ids"][idx]
                doc = all_records["documents"][idx] if all_records["documents"] else ""
                meta = all_records["metadatas"][idx] if all_records["metadatas"] else {}
                
                age_stage = meta.get("age_stage", "unknown")
                memory_type = meta.get("memory_type", "interaction")
                emotional_tag = meta.get("emotional_tag", "none")
                importance = meta.get("importance", 0.5)
                timestamp = meta.get("timestamp", 0.0)

                # Filter logic
                if filter_age and filter_age != "All" and age_stage != filter_age:
                    continue
                if filter_type and filter_type != "All" and memory_type != filter_type:
                    continue

                rows.append([
                    mem_id,
                    doc,
                    age_stage,
                    memory_type,
                    emotional_tag,
                    round(importance, 2),
                    timestamp
                ])
            
            # Sort by timestamp descending
            rows.sort(key=lambda x: x[6], reverse=True)
            
            status_text = f"Loaded {len(rows)} / {count} memories from '{collection_name}'."
            return rows, status_text
        except Exception as e:
            return [], f"Error loading database: {str(e)}"

    def get_stats():
        try:
            _, collection = get_chroma_client_and_collection(persist_dir, collection_name)
            count = collection.count()
            if count == 0:
                return "Collection is empty.", "N/A", "N/A", "N/A"
            
            all_records = collection.get()
            types = {}
            ages = {}
            emotions = {}
            
            for meta in all_records["metadatas"] or []:
                m_type = meta.get("memory_type", "unknown")
                age = meta.get("age_stage", "unknown")
                emotion = meta.get("emotional_tag", "none")
                
                types[m_type] = types.get(m_type, 0) + 1
                ages[age] = ages.get(age, 0) + 1
                emotions[emotion] = emotions.get(emotion, 0) + 1

            type_summary = ", ".join([f"{k}: {v}" for k, v in types.items()])
            age_summary = ", ".join([f"{k}: {v}" for k, v in ages.items()])
            emotion_summary = ", ".join([f"{k}: {v}" for k, v in emotions.items()])

            return f"{count} Memories", type_summary, age_summary, emotion_summary
        except Exception as e:
            return f"Error: {e}", "N/A", "N/A", "N/A"

    def semantic_search(query_text, num_results):
        if not query_text.strip():
            return [], "Please enter a search query."
        try:
            _, collection = get_chroma_client_and_collection(persist_dir, collection_name)
            count = collection.count()
            if count == 0:
                return [], "Database is empty."

            # Query ChromaDB
            results = collection.query(
                query_texts=[query_text],
                n_results=min(int(num_results), count)
            )

            rows = []
            if results and results["ids"] and results["ids"][0]:
                for idx, mem_id in enumerate(results["ids"][0]):
                    doc = results["documents"][0][idx] if results["documents"] else ""
                    meta = results["metadatas"][0][idx] if results["metadatas"] else {}
                    distance = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0
                    
                    # Convert distance to similarity score
                    similarity = round((1.0 - distance) * 100, 1)
                    
                    age_stage = meta.get("age_stage", "unknown")
                    memory_type = meta.get("memory_type", "interaction")
                    emotional_tag = meta.get("emotional_tag", "none")
                    importance = meta.get("importance", 0.5)
                    timestamp = meta.get("timestamp", 0.0)

                    rows.append([
                        mem_id,
                        doc,
                        f"{similarity}% (dist: {round(distance, 3)})",
                        age_stage,
                        memory_type,
                        emotional_tag,
                        round(importance, 2),
                        timestamp
                    ])
            
            return rows, f"Found {len(rows)} matching results for query: '{query_text}'"
        except Exception as e:
            return [], f"Search failed: {str(e)}"

    def get_details(selected_row, evt: gr.SelectData):
        # selected_row is the full table, evt contains the index clicked
        try:
            row_idx = evt.index[0]
            val = selected_row[row_idx]
            
            detail_md = f"""
### Memory Details

**ID:** `{val[0]}`
**Age Stage:** `{val[2]}`
**Type:** `{val[3]}`
**Emotion:** `{val[4]}`
**Importance:** `{val[5]}`
**Timestamp:** `{val[6]}`

---

#### Document Content:
```text
{val[1]}
```
"""
            return detail_md
        except Exception as e:
            return f"Error displaying details: {e}"

    def add_test_memory(content, age_stage, memory_type, emotional_tag, importance):
        if not content.strip():
            return "Content cannot be empty."
        try:
            import uuid
            import time
            _, collection = get_chroma_client_and_collection(persist_dir, collection_name)
            
            mem_id = str(uuid.uuid4())
            metadata = {
                "age_stage": age_stage,
                "memory_type": memory_type,
                "emotional_tag": emotional_tag if emotional_tag else "none",
                "importance": float(importance),
                "timestamp": time.time()
            }
            
            collection.add(
                ids=[mem_id],
                documents=[content],
                metadatas=[metadata]
            )
            return f"Successfully added test memory with ID {mem_id}"
        except Exception as e:
            return f"Failed to add memory: {e}"

    def delete_memory(memory_id):
        if not memory_id.strip():
            return "Please provide a valid Memory ID."
        try:
            _, collection = get_chroma_client_and_collection(persist_dir, collection_name)
            collection.delete(ids=[memory_id])
            return f"Successfully deleted memory ID: {memory_id}"
        except Exception as e:
            return f"Failed to delete memory: {e}"

    def export_database():
        try:
            _, collection = get_chroma_client_and_collection(persist_dir, collection_name)
            count = collection.count()
            if count == 0:
                return "{}", "Database is empty, nothing to export."
            
            all_records = collection.get()
            export_data = []
            for idx in range(len(all_records["ids"])):
                export_data.append({
                    "id": all_records["ids"][idx],
                    "document": all_records["documents"][idx] if all_records["documents"] else "",
                    "metadata": all_records["metadatas"][idx] if all_records["metadatas"] else {}
                })
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            filename = project_root / "data" / "memory" / f"chroma_export_{int(time.time())}.json"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_str)
                
            return json_str, f"Database exported successfully to:\n{filename}"
        except Exception as e:
            return "{}", f"Export failed: {e}"

    # Custom styling theme
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="indigo",
        neutral_hue="slate"
    )

    with gr.Blocks(title="ChromaDB Memory Explorer") as demo:
        gr.Markdown(
            f"""
            # 🧠 ChromaDB Memory Explorer
            Explore, search, and manage semantic memories stored in the AI Actor database.
            
            * **Database Path:** `{persist_dir}`
            * **Active Collection:** `{collection_name}`
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                stat_count = gr.Textbox(label="Total Memories", value="Calculating...", interactive=False)
            with gr.Column(scale=1):
                stat_types = gr.Textbox(label="Memory Types Breakdown", value="Calculating...", interactive=False)
            with gr.Column(scale=1):
                stat_ages = gr.Textbox(label="Age Stages Breakdown", value="Calculating...", interactive=False)
            with gr.Column(scale=1):
                stat_emotions = gr.Textbox(label="Emotions Breakdown", value="Calculating...", interactive=False)

        def update_stats():
            c, t, a, e = get_stats()
            return c, t, a, e

        with gr.Tab("📋 View & Filter"):
            with gr.Row():
                age_filter = gr.Dropdown(
                    choices=["All", "10-15", "15-20", "20-25", "25-30", "30-40", "40-50", "50-60", "60-70"],
                    value="All",
                    label="Filter by Age Stage"
                )
                type_filter = gr.Dropdown(
                    choices=["All", "interaction", "thought", "summary", "fact"],
                    value="All",
                    label="Filter by Memory Type"
                )
                refresh_btn = gr.Button("🔄 Refresh Database", variant="primary")
            
            db_status = gr.Markdown("Loading data...")
            
            # Columns: ID, Document, Age, Type, Emotion, Importance, Timestamp
            db_table = gr.Dataframe(
                headers=["ID", "Document Content", "Age Stage", "Memory Type", "Emotion", "Importance", "Timestamp"],
                datatype=["str", "str", "str", "str", "str", "number", "number"],
                column_count=(7, "fixed"),
                interactive=False,
                label="Click any row below to view full details:"
            )

            detail_view = gr.Markdown("*Select a memory row above to show details here.*")

            # Link selection
            db_table.select(fn=get_details, inputs=[db_table], outputs=[detail_view])

            # Trigger loading
            refresh_btn.click(
                fn=load_data,
                inputs=[age_filter, type_filter],
                outputs=[db_table, db_status]
            ).then(
                fn=update_stats,
                outputs=[stat_count, stat_types, stat_ages, stat_emotions]
            )

            # Auto-load on show
            demo.load(
                fn=load_data,
                inputs=[age_filter, type_filter],
                outputs=[db_table, db_status]
            ).then(
                fn=update_stats,
                outputs=[stat_count, stat_types, stat_ages, stat_emotions]
            )
            
            # Dropdown trigger reload automatically
            age_filter.change(fn=load_data, inputs=[age_filter, type_filter], outputs=[db_table, db_status])
            type_filter.change(fn=load_data, inputs=[age_filter, type_filter], outputs=[db_table, db_status])

        with gr.Tab("🔍 Semantic Search"):
            gr.Markdown("Query ChromaDB using vector similarity search to find memories conceptually closest to your query text.")
            with gr.Row():
                search_query = gr.Textbox(placeholder="Enter search query...", label="Query Text")
                search_limit = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Results Limit")
                search_btn = gr.Button("🔍 Semantic Search", variant="primary")
            
            search_status = gr.Markdown("Enter a query to start search.")
            
            # Columns: ID, Document, Similarity, Age, Type, Emotion, Importance, Timestamp
            search_table = gr.Dataframe(
                headers=["ID", "Document Content", "Similarity Score", "Age Stage", "Memory Type", "Emotion", "Importance", "Timestamp"],
                datatype=["str", "str", "str", "str", "str", "str", "number", "number"],
                column_count=(8, "fixed"),
                interactive=False
            )
            
            search_detail = gr.Markdown("*Select a row below to show details here.*")
            search_table.select(fn=get_details, inputs=[search_table], outputs=[search_detail])

            search_btn.click(
                fn=semantic_search,
                inputs=[search_query, search_limit],
                outputs=[search_table, search_status]
            )

        with gr.Tab("🛠️ Operations & Debug"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ➕ Add Test Memory")
                    add_content = gr.Textbox(label="Memory Document Content", placeholder="Type memory content here...", lines=3)
                    with gr.Row():
                        add_age = gr.Dropdown(choices=["10-15", "15-20", "20-25", "25-30", "30-40", "40-50", "50-60", "60-70"], value="10-15", label="Age Stage")
                        add_type = gr.Dropdown(choices=["interaction", "thought", "summary", "fact"], value="interaction", label="Memory Type")
                        add_emotion = gr.Textbox(label="Emotional Tag (e.g. alegre, dudosa, triste)", value="neutral")
                        add_importance = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Importance")
                    add_btn = gr.Button("Add Memory", variant="secondary")
                    add_status = gr.Textbox(label="Status", interactive=False)
                    
                    add_btn.click(
                        fn=add_test_memory,
                        inputs=[add_content, add_age, add_type, add_emotion, add_importance],
                        outputs=[add_status]
                    )

                with gr.Column():
                    gr.Markdown("### ❌ Delete Memory")
                    del_id = gr.Textbox(label="Memory ID to Delete", placeholder="Paste UUID here...")
                    del_btn = gr.Button("Delete Memory", variant="stop")
                    del_status = gr.Textbox(label="Status", interactive=False)
                    
                    del_btn.click(
                        fn=delete_memory,
                        inputs=[del_id],
                        outputs=[del_status]
                    )

            gr.HTML("<hr>")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 💾 Export Database")
                    gr.Markdown("Export all database records into a JSON backup file stored in your `data/memory/` directory.")
                    export_btn = gr.Button("Backup & Export to JSON", variant="secondary")
                with gr.Column():
                    export_status = gr.Textbox(label="Status", interactive=False)
            
            export_json = gr.Code(label="Exported JSON Data", language="json")
            
            export_btn.click(
                fn=export_database,
                outputs=[export_json, export_status]
            )

    # Launch server
    demo.launch(server_name="127.0.0.1", server_port=port, share=False, theme=theme)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB Memory Explorer for AI Actor Project")
    parser.add_argument("--cli", action="store_true", help="Run in command-line interface mode using Rich")
    parser.add_argument("--port", type=int, default=7860, help="Gradio web server port (default: 7860)")
    parser.add_argument("--dir", type=str, default=db_config["persist_dir"], help="Path to ChromaDB persist directory")
    parser.add_argument("--collection", type=str, default=db_config["collection_name"], help="ChromaDB collection name")
    
    args = parser.parse_args()
    
    # Import time for timestamp operations
    import time

    if args.cli:
        run_cli(args.dir, args.collection)
    else:
        run_web_ui(args.dir, args.collection, args.port)
