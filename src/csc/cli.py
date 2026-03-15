"""CLI entry point for the Clinical Supply Chain modeling system."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Clinical Supply Chain Multi-Agent Modeling System.

    Model the end-to-end clinical supply chain using autonomous agents.

    Three methods available (set CSC_METHOD in .env or use --method flag):

    \b
    LLM agents (default): 5 LLM-powered agents via LiteLLM
    RL agents:            4 PPO-trained reinforcement learning agents
    BDI agents:           5 rule-based Belief-Desire-Intention agents
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.option("--trials", default=15, help="Number of clinical trials to generate")
@click.option("--sites", default=30, help="Number of clinical sites")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--output-dir", default="src/csc/data/output", type=click.Path(), help="Output directory for data files")
def generate(trials: int, sites: int, seed: int, output_dir: str) -> None:
    """Generate synthetic clinical supply chain data."""
    from csc.data.master_generator import MasterGenerator

    out = Path(output_dir)
    console.print(f"[bold]Generating synthetic data[/] (seed={seed}, sites={sites})")

    gen = MasterGenerator(seed=seed, num_sites=sites)
    gen.generate()
    gen.save(out)

    console.print(f"\n[green]Generated {len(gen.trials)} trials, {len(gen.sites)} sites[/]")
    console.print(f"[green]Materials: {len(gen.materials.drug_substances)} DS, "
                  f"{len(gen.materials.drug_products)} DP, "
                  f"{len(gen.materials.primary_packs)} PP, "
                  f"{len(gen.materials.finished_goods)} FG[/]")
    console.print(f"[green]Enrollment forecasts: {len(gen.enrollment_forecasts)}[/]")
    console.print(f"[green]Data saved to {out.resolve()}[/]")


@main.command()
@click.option("--agent", "agent_name", default=None, help="Run a specific agent")
@click.option("--all", "run_all", is_flag=True, help="Run the full pipeline")
@click.option("--data-dir", default="src/csc/data/output", type=click.Path(exists=True), help="Path to synthetic data")
@click.option("--output-dir", default="reports", type=click.Path(), help="Where to write reports")
@click.option("--model", default=None, help="Claude model to use (overrides .env, LLM method only)")
@click.option("--method", default=None, type=click.Choice(["llm", "rl", "bdi"]), help="Agent method: llm, rl, or bdi (overrides .env CSC_METHOD)")
@click.option("--format", "fmt", default="both", type=click.Choice(["csv", "json", "both"]), help="Report format")
def run(agent_name: str | None, run_all: bool, data_dir: str, output_dir: str, model: str | None, method: str | None, fmt: str) -> None:
    """Run agents (full pipeline or individual).

    Use --method to choose between LLM-based agents and RL-based agents.
    """
    from csc.config import Config
    from csc.orchestrator.resolver import resolve_conflicts
    from csc.reports.terminal import print_summary
    from csc.reports.writer import write_reports

    if not run_all and not agent_name:
        console.print("[red]Specify --all or --agent <name>[/]")
        raise SystemExit(1)

    config = Config.from_env()
    active_method = method or config.method

    if active_method == "rl":
        from csc.orchestrator.rl_pipeline import RLSupplyChainPipeline

        pipeline = RLSupplyChainPipeline(config)
        pipeline.load_data(Path(data_dir))

        if run_all:
            pipeline.run_full()
            resolve_conflicts(pipeline.state)
        else:
            pipeline.run_agent(agent_name)
    elif active_method == "bdi":
        from csc.orchestrator.bdi_pipeline import BDISupplyChainPipeline

        pipeline = BDISupplyChainPipeline(config)
        pipeline.load_data(Path(data_dir))

        if run_all:
            pipeline.run_full()
            resolve_conflicts(pipeline.state)
        else:
            pipeline.run_agent(agent_name)
    else:
        from csc.orchestrator.pipeline import SupplyChainPipeline

        if model:
            config = Config(
                model=model,
                max_agent_turns=config.max_agent_turns,
                anthropic_api_key=config.anthropic_api_key,
                openai_api_key=config.openai_api_key,
                gemini_api_key=config.gemini_api_key,
                nebius_api_key=config.nebius_api_key,
            )

        pipeline = SupplyChainPipeline(config)
        pipeline.load_data(Path(data_dir))

        if run_all:
            pipeline.run_full()
            resolve_conflicts(pipeline.state)
        else:
            pipeline.run_agent(agent_name)

    # Write reports
    out = Path(output_dir)
    created = write_reports(pipeline.state, out, fmt)
    console.print(f"\n[bold]Reports written to {out.resolve()}:[/]")
    for f in created:
        console.print(f"  {f.name}")

    # Print terminal summary
    print_summary(pipeline.state)


@main.command()
@click.option("--agent", "agent_name", default=None, help="Train a specific RL agent (demand_forecast, capacity_allocation, batch_scheduling, inventory_safety_stock)")
@click.option("--timesteps", default=None, type=int, help="Total training timesteps (overrides .env)")
@click.option("--seed", default=None, type=int, help="Training random seed (overrides .env)")
def train(agent_name: str | None, timesteps: int | None, seed: int | None) -> None:
    """Train RL agents on synthetic supply chain data.

    Trains one or all RL agents using PPO. Models are saved to CSC_RL_MODEL_DIR.
    Requires the RL dependencies: pip install -e ".[rl]"
    """
    from csc.config import Config
    from csc.rl.training.trainer import RLTrainer

    config = Config.from_env()
    trainer = RLTrainer(config)
    trainer.train(agent_name=agent_name, total_timesteps=timesteps, seed=seed)


@main.command()
@click.option("--data-dir", default="src/csc/data/output", type=click.Path(exists=True), help="Path to data")
@click.option("--entity", type=click.Choice(["trials", "sites", "depots", "plants", "materials", "enrollment"]), help="Entity to inspect")
def inspect(data_dir: str, entity: str) -> None:
    """Inspect generated synthetic data."""
    from csc.orchestrator.state import SharedState
    from rich.table import Table

    state = SharedState()
    state.load_from_dir(Path(data_dir))

    if entity == "trials":
        table = Table(title="Clinical Trials")
        table.add_column("Protocol", style="cyan")
        table.add_column("Therapy Area")
        table.add_column("Phase")
        table.add_column("Enrollment", justify="right")
        table.add_column("Sites", justify="right")
        table.add_column("FSFV")
        table.add_column("LSLV")
        for t in state.trials:
            table.add_row(t.protocol_number, t.therapy_area.value, t.phase.value,
                         str(t.planned_enrollment), str(len(t.sites)), str(t.fsfv), str(t.lslv))
        console.print(table)

    elif entity == "sites":
        table = Table(title="Clinical Sites")
        table.add_column("Name", style="cyan")
        table.add_column("Region")
        table.add_column("Country")
        table.add_column("City")
        table.add_column("Max Patients", justify="right")
        for s in state.sites:
            table.add_row(s.name, s.region.value, s.country, s.city, str(s.max_patients))
        console.print(table)

    elif entity == "depots":
        table = Table(title="Packing Depots")
        table.add_column("Name", style="cyan")
        table.add_column("Region")
        table.add_column("Type")
        table.add_column("Pkg Lines", justify="right")
        table.add_column("Lbl Lines", justify="right")
        table.add_column("Languages")
        for d in state.depots:
            table.add_row(d.name, d.region.value, d.depot_type,
                         str(d.packaging_lines), str(d.labeling_lines), ", ".join(d.supported_languages))
        console.print(table)

    elif entity == "plants":
        table = Table(title="Pilot Plants")
        table.add_column("Name", style="cyan")
        table.add_column("Region")
        table.add_column("Lines", justify="right")
        table.add_column("Capacity (kg/yr)", justify="right")
        for p in state.plants:
            table.add_row(p.name, p.region.value, str(p.equipment_lines), f"{p.annual_capacity_kg:,.0f}")
        console.print(table)

    elif entity == "materials":
        console.print(f"\n[bold]Drug Substances: {len(state.materials.drug_substances)}[/]")
        for ds in state.materials.drug_substances:
            console.print(f"  {ds.name} | {ds.batch_size_kg}kg batch | yield {ds.yield_rate:.0%} | {ds.storage_condition}")

        console.print(f"\n[bold]Drug Products: {len(state.materials.drug_products)}[/]")
        for dp in state.materials.drug_products:
            console.print(f"  {dp.name} | {dp.formulation_type.value} | {dp.batch_size_units} units/batch | yield {dp.yield_rate:.0%}")

        console.print(f"\n[bold]Primary Packs: {len(state.materials.primary_packs)}[/]")
        console.print(f"[bold]Finished Goods: {len(state.materials.finished_goods)}[/]")

    elif entity == "enrollment":
        console.print(f"\n[bold]Enrollment Forecasts: {len(state.enrollment_forecasts)} records[/]")
        # Show summary per trial
        trial_ids = set(f.trial_id for f in state.enrollment_forecasts)
        for tid in trial_ids:
            forecasts = [f for f in state.enrollment_forecasts if f.trial_id == tid]
            trial = next((t for t in state.trials if t.id == tid), None)
            if trial and forecasts:
                max_enrolled = max(f.cumulative_enrolled for f in forecasts)
                months = len(set(f.month for f in forecasts))
                console.print(f"  {trial.protocol_number}: {max_enrolled} enrolled over {months} months")


if __name__ == "__main__":
    main()
