#!/usr/bin/env python3
"""Production deployment script for FERC EQR Scraper with Turso integration."""

import os
import sys
import click
from typing import Optional

def check_turso_config() -> bool:
    """Check if Turso configuration is available."""
    return bool(os.environ.get("TURSO_DATABASE_URL") and os.environ.get("TURSO_AUTH_TOKEN"))

def validate_turso_connection() -> bool:
    """Validate Turso database connection."""
    try:
        from database import FERCDatabase
        
        with FERCDatabase() as db:
            db._test_connection()
            return True
    except Exception as e:
        click.echo(f"❌ Turso connection failed: {e}", err=True)
        return False

@click.command()
@click.option('--turso-url', help='Turso database URL (or set TURSO_DATABASE_URL env var)')
@click.option('--turso-token', help='Turso auth token (or set TURSO_AUTH_TOKEN env var)')
@click.option('--init-schema', is_flag=True, default=False, help='Initialize production schema')
@click.option('--test-connection', is_flag=True, default=False, help='Test database connection only')
def deploy_production(turso_url: Optional[str], turso_token: Optional[str], 
                     init_schema: bool, test_connection: bool) -> None:
    """Deploy FERC EQR Scraper to production with Turso database."""
    
    click.echo("🚀 FERC EQR Scraper - Production Deployment")
    click.echo("=" * 50)
    
    # Set environment variables if provided
    if turso_url:
        os.environ["TURSO_DATABASE_URL"] = turso_url
    if turso_token:
        os.environ["TURSO_AUTH_TOKEN"] = turso_token
    
    # Check configuration
    if not check_turso_config():
        click.echo("❌ Turso configuration missing!", err=True)
        click.echo("\nPlease set the following environment variables:")
        click.echo("  export TURSO_DATABASE_URL='libsql://your-database.turso.io'")
        click.echo("  export TURSO_AUTH_TOKEN='your-auth-token'")
        click.echo("\nOr use the command line options:")
        click.echo("  python deploy_production.py --turso-url 'libsql://your-db.turso.io' --turso-token 'your-token'")
        sys.exit(1)
    
    click.echo(f"✅ Turso URL: {os.environ['TURSO_DATABASE_URL']}")
    click.echo("✅ Turso token: [REDACTED]")
    
    # Test connection
    click.echo("\n📡 Testing database connection...")
    if not validate_turso_connection():
        sys.exit(1)
    click.echo("✅ Database connection successful!")
    
    if test_connection:
        click.echo("✅ Connection test completed successfully!")
        return
    
    # Initialize schema if requested
    if init_schema:
        click.echo("\n🏗️  Initializing production schema...")
        try:
            from database import FERCDatabase
            
            with FERCDatabase() as db:
                db.initialize_production_schema()
                
            click.echo("✅ Production schema initialized successfully!")
            
            # Show schema info
            with FERCDatabase() as db:
                info = db.get_existing_data_info()
                click.echo(f"📊 Database contains {info['total_tables']} tables with {info['total_rows']} total rows")
                
        except Exception as e:
            click.echo(f"❌ Schema initialization failed: {e}", err=True)
            sys.exit(1)
    
    # Production recommendations
    click.echo("\n🎯 Production Setup Complete!")
    click.echo("\nRecommended next steps:")
    click.echo("1. Set up your data processing pipeline:")
    click.echo("   python main.py --start-year 2024 --end-year 2024")
    click.echo("\n2. Monitor memory usage during processing:")
    click.echo("   export MAX_MEMORY_USAGE_PCT=60  # Lower for production")
    click.echo("\n3. Configure logging for production:")
    click.echo("   export LOG_LEVEL=INFO")
    click.echo("\n4. Set up regular data updates with cron or similar")
    
    click.echo("\n🎉 Your FERC scraper is production-ready with Turso!")

if __name__ == "__main__":
    deploy_production()