#!/usr/bin/env python3
import os
import sys
import re
import argparse
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dotenv import load_dotenv
from github import Github
from openai import OpenAI

load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate weekly status summaries for engineering teams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/weekly_status_gen.py                    # Concise summaries (highlights only)
  python src/weekly_status_gen.py --detailed         # Include detailed PR summaries  
  python src/weekly_status_gen.py --teams voyager    # Specific team only
  python src/weekly_status_gen.py --days 14          # Look back 14 days
  python src/weekly_status_gen.py --detailed --days 3 --teams apollo,sovereign
        """
    )
    
    parser.add_argument(
        '--detailed', 
        action='store_true', 
        help='Include detailed PR summaries'
    )
    
    parser.add_argument(
        '--teams', 
        help='Comma-separated list of teams to process (ie: voyager,forte,sovereign,apollo)'
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=6, 
        help='Number of days to look back for PRs (default: 7)'
    )
    
    return parser.parse_args()

def extract_tag(pr_title, pr_files=None, team_name=None):
    """Extract subteam tag from PR title with smart filtering."""
    
    VALID_SUBTEAMS = {
        'sovereign': {'BDC', 'COPS', 'IRMAFE', 'MARCOM', 'PMS', 'RENGP', 'SRS', 
                     'STUDIOCORE', 'STUDIOFE', 'TPMO', 'WALLE', 'NEXUS', 'CSC'},
        'voyager': {'VOYAGER'},
        'forte': {'FORTE'},
        'apollo': {'APOLLO'}
    }
    
    valid_tags = VALID_SUBTEAMS.get(team_name, set())
    title_upper = pr_title.upper()
    
    pattern = r'\b([A-Z]+)[-\s]?\d+'
    matches = re.findall(pattern, title_upper)
    
    for match in matches:
        if valid_tags and match in valid_tags:
            return match
        elif not valid_tags and len(match) >= 3 and match not in {'BUG', 'SET', 'FIX', 'FEAT', 'CHORE', 'TEST', 'CONNECT'}:
            return match
    
    for tag in valid_tags:
        if tag in title_upper:
            return tag
    
    if pr_files and valid_tags:
        for file_path in pr_files:
            path_upper = file_path.upper()
            for tag in valid_tags:
                # check for /tag/ or -tag- patterns in path
                if f'/{tag}/' in path_upper or f'-{tag}-' in path_upper or f'_{tag}_' in path_upper:
                    return tag
                # special cases for common patterns
                if tag == 'PMS' and '/pms' in file_path.lower():
                    return 'PMS'
                if tag == 'COPS' and '/cops' in file_path.lower():
                    return 'COPS'
                if tag == 'STUDIOCORE' and 'studio' in file_path.lower() and 'core' in file_path.lower():
                    return 'STUDIOCORE'
    
    return "UNTAGGED"

def group_prs_by_tag(prs, team_name=None):
    """Group PRs by their subteam tags."""
    grouped = defaultdict(list)
    for pr in prs:
        tag = extract_tag(pr['title'], pr.get('files', []), team_name)
        grouped[tag].append(pr)
    return grouped

def test_github_connection():
    token = os.getenv('GITHUB_TOKEN')
    
    g = Github(token)
    user = g.get_user()
    print(f"GitHub: Connected as {user.login}")
    return g

def test_llm_connections():
    """Test OpenAI API connection at startup."""
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        print("OpenAI: No API key found")
        return None
        
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5
    )
    print("OpenAI: Connected successfully")
    return 'openai'
    
    
def collect_prs(repo_name, directory_path, days_back=6):
    """Most efficient: Find commits to directory, then get their PRs."""
    token = os.getenv('GITHUB_TOKEN')
    g = Github(token)
    repo = g.get_repo(repo_name)
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    team_data = {
        'directory': directory_path,
        'merged_prs': [],
        'stats': defaultdict(int)
    }
    
    print(f"\nGetting commits to src/{directory_path}/ since {cutoff_date.date()}")
    
    # get commits that touched the directory
    commits = repo.get_commits(
        since=cutoff_date,
        path=f"src/{directory_path}/"
    )
    
    pr_numbers = set()
    commit_count = 0
    
    for commit in commits:
        commit_count += 1
            
        for pr in commit.get_pulls():
            pr_numbers.add(pr.number)
            
    for pr_num in pr_numbers:
        
        pr = repo.get_pull(pr_num)
        
        files_in_dir = []
        for file in pr.get_files():
            if file.filename.startswith(f"src/{directory_path}/"):
                files_in_dir.append(file.filename)
                team_data['stats']['files_changed'] += 1
                team_data['stats']['additions'] += file.additions
                team_data['stats']['deletions'] += file.deletions
        
        pr_info = {
            'number': pr.number,
            'title': pr.title,
            'author': pr.user.login,
            'files': files_in_dir,
            'additions': pr.additions,
            'deletions': pr.deletions,
            'description': pr.body[:500] if pr.body else ""
        }
        
        # check the actual PR status - only include merged PRs
        if pr.state == 'closed' and pr.merged and pr.merged_at:
            pr_info['merged_at'] = pr.merged_at
            team_data['merged_prs'].append(pr_info)
       
    
    team_data['stats']['merged_count'] = len(team_data['merged_prs'])
    
    return team_data

def generate_summary(team_data, llm_provider='openai', detailed=True):
    """Generate a business-readable summary using LLM."""
    
    # prepare the context
    merged_prs_text = ""
    for pr in team_data['merged_prs']:
        merged_prs_text += f"- PR #{pr['number']}: {pr['title']} (by {pr['author']})\n"
        if pr['description']:
            merged_prs_text += f"  Description: {pr['description']}\n"
    
    # calculate total PRs for dynamic formatting
    total_prs = len(team_data['merged_prs'])
    
    # build prompt based on detailed flag
    if detailed:
        sections_instruction = """Create a concise technical summary with TWO sections:

                                PR Summaries:
                                List each PR with a one-sentence technical description of what was actually done.
                                Format exactly as shown (no markdown, no asterisks):
                                - PR #50326: Resolved circular dependency between auth and user modules
                                - PR #50327: Implemented parallel processing for bulk trade confirmations

                                Rules for PR summaries:
                                - Be specific about WHAT was done (e.g., "Added Redis caching to trade API" not "improved performance")
                                - Use technical terms (API, database, cache, queue, webhook, etc.)
                                - Mention specific components/services affected
                                - Keep each summary under 15 words

                                Weekly Highlights:"""
        output_instruction = "Output both PR Summaries and Weekly Highlights sections."
    else:
        sections_instruction = """Create a concise technical summary with ONE section:

Weekly Highlights:"""
        output_instruction = "Output only the Weekly Highlights section. Do NOT include PR Summaries."
    
    if total_prs <= 8:
        bullet_instruction = "Write exactly 3-4 bullet points summarizing the key deliverables/changes. Combine multiple related PRs into single bullets."
        format_instruction = """Format with + symbol:
                            + Added Redis caching layer for trade confirmations
                            + Migrated payment reconciliation from batch to real-time processing via Kafka streams
                            + Fixed critical bug in margin calculation affecting portfolio valuation"""
    elif total_prs <= 15:
        bullet_instruction = "Write exactly 4-5 bullet points summarizing the key deliverables/changes. MAXIMUM 5 bullets. Combine multiple related PRs into single bullets."
        format_instruction = """Format with + symbol:
                            + Added Redis caching layer for trade confirmations
                            + Migrated payment reconciliation from batch to real-time processing via Kafka streams
                            + Fixed critical bug in margin calculation affecting portfolio valuation"""
    else:
        bullet_instruction = f"Write exactly 5-6 bullet points with sub-categories for this large team ({total_prs} PRs). MAXIMUM 6 bullets. Group multiple related PRs under main topics."
        format_instruction = """Format with + symbol and sub-bullets for categories:
                            + API & Backend Changes:
                            - Added new endpoints for document management and status updates
                            - Implemented Kafka message processing for real-time data sync
                            + UI & Frontend Updates:
                            - Fixed visibility issues with approval buttons in management interface  
                            - Updated form validations and field requirements
                            + Bug Fixes & Performance:
                            - Resolved null pointer exceptions in synchronization processes
                            - Fixed configuration issues affecting service deployments"""
    
    prompt = f"""Analyze the GitHub activity for the {team_data['directory']} team from the past week.

                INPUT DATA:
                ===========
                MERGED PRs:
                {merged_prs_text}

                YOUR TASK:
                ==========
                {sections_instruction}
                {bullet_instruction}
                {format_instruction}

                Rules for highlights:
                - State ONLY what was built/fixed/changed - NO business impact phrases
                - Include technical details (technologies, systems, components)
                - Each bullet/sub-bullet should be 8-15 words maximum
                - FORBIDDEN phrases: "improving", "enhancing", "streamlining", "enabling", "ensuring", "optimizing"
                - FORBIDDEN suffixes: "for better X", "to improve Y", "enabling Z"
                - Just state the technical action taken
                - CRITICAL: Combine multiple related PRs into single bullets - do NOT list every individual change
                - Group similar changes together (e.g., "Fixed multiple permission bugs" instead of listing each bug)
                - For large teams: group related PRs under logical categories (API, UI, Infrastructure, Bug Fixes, etc.)
                - Prioritize the most impactful changes - minor config updates can be grouped or omitted

                {output_instruction} No additional commentary."""

    if llm_provider != 'openai':
        return "No LLM provider available"
        
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.8
    )
    return response.choices[0].message.content

def generate_summary_for_large_team(team_data, llm_provider='openai', max_prs_for_single_summary=40, detailed=True):
    """Generate summaries for large teams, grouped by subteam tags."""
    total_prs = len(team_data['merged_prs'])
    
    # for small teams, use single summary
    if total_prs <= max_prs_for_single_summary:
        return generate_summary(team_data, llm_provider, detailed)
    
    print(f"\n=== {team_data['directory'].upper()} TEAM ({total_prs} PRs) - Splitting directory into subteams ===")
    
    merged_groups = group_prs_by_tag(team_data['merged_prs'], team_data['directory'])
    
    all_tags = sorted(merged_groups.keys())
    
    # show all subteams that will be processed
    subteam_info = []
    for tag in all_tags:
        merged_count = len(merged_groups.get(tag, []))
        if merged_count > 0:
            subteam_info.append(f"{tag}({merged_count})")
    
    print(f"Subteams to process: {', '.join(subteam_info)}")
    print()
    
    summaries = {}
    for i, tag in enumerate(all_tags, 1):
        subteam_data = {
            'directory': f"{team_data['directory']}/{tag}",
            'merged_prs': merged_groups.get(tag, []),
            'stats': defaultdict(int)
        }
        
        # Calculate stats for subteam
        for pr in subteam_data['merged_prs']:
            subteam_data['stats']['merged_count'] += 1
            subteam_data['stats']['files_changed'] += len(pr.get('files', []))
            subteam_data['stats']['additions'] += pr.get('additions', 0)
            subteam_data['stats']['deletions'] += pr.get('deletions', 0)
            
        # Only summarize if subteam has PRs
        if subteam_data['merged_prs']:
            total_subteam_prs = len(subteam_data['merged_prs'])
            print(f"[{i}/{len(all_tags)}] Processing {tag} ({total_subteam_prs} PRs)...", end=" ", flush=True)
            summaries[tag] = generate_summary(subteam_data, llm_provider, detailed)
            print("✓")
    
    return summaries

def main():
    args = parse_args()
    
    github_client = test_github_connection()
    if not github_client:
        print("Error: GitHub connection failed")
        sys.exit(1)
    
    llm_provider = test_llm_connections()
        
    repo = "clear-street-internal/fleet"
    teams = args.teams.split(',')
    
    for team in teams:
        data = collect_prs(repo, team, days_back=args.days)
        
        total_prs = len(data['merged_prs'])
        
        if llm_provider and data['merged_prs']:
            summary = generate_summary_for_large_team(data, llm_provider, detailed=args.detailed)
            
            # Handle single summary (small teams)
            if isinstance(summary, str):
                print(f"\n=== {team.upper()} TEAM WEEKLY SUMMARY ({total_prs} PRs) ===")
                print(summary)
                print("=" * 40)
            
            # Handle multiple summaries (large teams with subteams)
            elif isinstance(summary, dict):
                # Calculate PR counts for each subteam
                merged_groups = group_prs_by_tag(data['merged_prs'], team)
                
                for tag, sub_summary in summary.items():
                    merged_count = len(merged_groups.get(tag, []))
                    subteam_total = merged_count
                    
                    print(f"\n=== {team.upper()}/{tag} SUMMARY ({subteam_total} PRs) ===")
                    print(sub_summary)
                    print("-" * 40)
    
if __name__ == "__main__":
    main()
