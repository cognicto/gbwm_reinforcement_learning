"""
Interactive inference with trained GBWM model

Usage:
    python experiments/run_inference.py --model_path data/results/experiment_name/final_model_safe.pth
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
import json

from src.evaluation.evaluator import GBWMEvaluator
from src.environment.gbwm_env import make_gbwm_env


def interactive_session(evaluator, num_goals=4):
    """Run interactive inference session"""

    print("\n🎯 Interactive GBWM Financial Advisor")
    print("=" * 60)
    print("Enter your current situation to get AI-powered financial advice!")
    print("Commands: 'quit' or 'q' to exit, 'help' for examples")
    print("=" * 60)

    # Show goal schedule
    if num_goals == 4:
        goal_years = [4, 8, 12, 16]
    elif num_goals == 2:
        goal_years = [8, 16]
    elif num_goals == 8:
        goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
    else:
        goal_years = list(range(1, 17))

    print(f"📅 Goals are available at years: {goal_years}")
    print("   (Years since starting your career at age 25)")
    print()

    while True:
        try:
            print("\n" + "=" * 40)
            print("📋 Enter Your Current Situation")
            print("=" * 40)

            # Get user input
            user_input = input("Your current age (25-40) or 'quit': ").strip()

            if user_input.lower() in ['quit', 'q', 'exit']:
                print("\n👋 Thank you for using GBWM Financial Advisor!")
                break

            if user_input.lower() == 'help':
                print("\n💡 Example scenarios:")
                print("  Age: 27, Wealth: $150,000 - Young professional")
                print("  Age: 33, Wealth: $250,000 - Mid-career")
                print("  Age: 38, Wealth: $500,000 - High earner")
                continue

            try:
                age = int(user_input)
                if not (25 <= age <= 40):
                    print("⚠️  Please enter age between 25-40")
                    continue
            except ValueError:
                print("⚠️  Please enter a valid age (number)")
                continue

            wealth_input = input("Current wealth ($): $").strip().replace('$', '').replace(',', '')

            try:
                wealth = float(wealth_input)
                if wealth < 0:
                    print("⚠️  Wealth should be positive")
                    continue
            except ValueError:
                print("⚠️  Please enter a valid wealth amount")
                continue

            # Process the input and get recommendation
            result = get_recommendation(evaluator, age, wealth, goal_years)
            display_recommendation(result, age, wealth)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again")


def get_recommendation(evaluator, age, wealth, goal_years):
    """Get AI recommendation for given age and wealth"""

    # Convert to model inputs
    time_step = age - 25  # Years since starting career
    normalized_time = time_step / 16
    normalized_wealth = min(wealth / 1000000, 1.0)  # Normalize to millions, cap at 1

    observation = np.array([normalized_time, normalized_wealth], dtype=np.float32)

    # Get AI recommendation
    action = evaluator.agent.predict(observation, deterministic=True)
    goal_action, portfolio_action = action

    # Get detailed analysis
    state_tensor = torch.FloatTensor(observation).unsqueeze(0)
    with torch.no_grad():
        goal_probs, portfolio_probs = evaluator.agent.policy_net(state_tensor)
        value_estimate = evaluator.agent.value_net(state_tensor)

    # Check if goal is available
    goal_available = time_step in goal_years
    goal_cost = None
    goal_utility = None
    can_afford = False

    if goal_available:
        goal_cost = 10 * (1.08 ** time_step)  # From paper formula
        goal_utility = 10 + time_step
        can_afford = wealth >= goal_cost

    return {
        'age': age,
        'wealth': wealth,
        'time_step': time_step,
        'goal_action': goal_action,
        'portfolio_action': portfolio_action,
        'goal_probs': goal_probs[0].cpu().numpy(),
        'portfolio_probs': portfolio_probs[0].cpu().numpy(),
        'value_estimate': value_estimate.item(),
        'goal_available': goal_available,
        'goal_cost': goal_cost,
        'goal_utility': goal_utility,
        'can_afford': can_afford
    }


def display_recommendation(result, age, wealth):
    """Display the AI recommendation in a user-friendly format"""

    print(f"\n🤖 AI Financial Advisor Recommendation")
    print("=" * 50)
    print(f"👤 Profile: {age} years old, ${wealth:,.0f} wealth")
    print()

    # Goal recommendation
    if result['goal_available']:
        print("🎯 GOAL ANALYSIS:")
        print(f"   📊 Goal Cost: ${result['goal_cost']:,.0f}")
        print(f"   🏆 Goal Value: {result['goal_utility']:.0f} utility points")
        print(f"   💰 Can Afford: {'✅ Yes' if result['can_afford'] else '❌ No'}")
        print()

        if result['goal_action'] == 1:
            confidence = result['goal_probs'][1] * 100
            print(f"💡 RECOMMENDATION: 🎯 TAKE THE GOAL!")
            print(f"   📈 Confidence: {confidence:.1f}%")
            if not result['can_afford']:
                print("   ⚠️  Note: You cannot afford this goal currently")
        else:
            confidence = result['goal_probs'][0] * 100
            print(f"💡 RECOMMENDATION: ⏳ SKIP THIS GOAL")
            print(f"   📈 Confidence: {confidence:.1f}%")

        print()
    else:
        print("🎯 No financial goal available at your current career stage")
        print()

    # Portfolio recommendation
    print("📈 INVESTMENT STRATEGY:")

    portfolio_num = result['portfolio_action'] + 1
    portfolio_confidence = result['portfolio_probs'][result['portfolio_action']] * 100

    # Categorize risk level
    if result['portfolio_action'] < 5:
        risk_level = "🛡️  CONSERVATIVE"
        risk_desc = "Focus on capital preservation"
    elif result['portfolio_action'] < 10:
        risk_level = "⚖️  MODERATE"
        risk_desc = "Balanced growth approach"
    else:
        risk_level = "🚀 AGGRESSIVE"
        risk_desc = "High growth potential"

    print(f"   🎯 Strategy: {risk_level}")
    print(f"   📊 Portfolio #{portfolio_num} of 15")
    print(f"   📈 Confidence: {portfolio_confidence:.1f}%")
    print(f"   💭 Approach: {risk_desc}")
    print()

    # Future outlook
    print("🔮 FUTURE OUTLOOK:")
    print(f"   🎯 Expected Total Utility: {result['value_estimate']:.1f} points")

    # Provide reasoning
    print("\n🧠 WHY THIS RECOMMENDATION?")

    if result['goal_available']:
        if result['goal_action'] == 1 and result['can_afford']:
            print("   ✅ You can afford this goal and it provides good value")
            print("   ✅ Taking it now aligns with optimal timing")
        elif result['goal_action'] == 1 and not result['can_afford']:
            print("   ⚠️  AI recommends this goal but you lack funds")
            print("   💡 Consider more aggressive investing to reach this goal")
        else:
            print("   💰 Saving money for more valuable future opportunities")
            print("   🎯 This goal may not justify the current cost")

    if result['portfolio_action'] < 5:
        print("   🛡️  Conservative approach: You have near-term goals or prefer safety")
    elif result['portfolio_action'] >= 10:
        print("   🚀 Aggressive approach: You have time horizon and risk tolerance")
    else:
        print("   ⚖️  Balanced approach: Steady growth with moderate risk")


def batch_inference(evaluator, scenarios):
    """Run batch inference on multiple scenarios"""

    print("\n📊 Batch Inference Results")
    print("=" * 60)

    results = []
    goal_years = [4, 8, 12, 16]  # Assume 4 goals

    for i, scenario in enumerate(scenarios):
        age, wealth = scenario['age'], scenario['wealth']
        result = get_recommendation(evaluator, age, wealth, goal_years)
        results.append(result)

        print(f"\n📋 Scenario {i + 1}: Age {age}, Wealth ${wealth:,}")
        print(f"🎯 Goal: {'Take' if result['goal_action'] == 1 else 'Skip'} "
              f"({result['goal_probs'][result['goal_action']]:.1%} confidence)")
        print(f"📈 Portfolio: #{result['portfolio_action'] + 1} "
              f"({result['portfolio_probs'][result['portfolio_action']]:.1%} confidence)")
        print(f"🔮 Expected Utility: {result['value_estimate']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Interactive inference with trained GBWM model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file')

    parser.add_argument('--num_goals', type=int, default=4,
                        choices=[1, 2, 4, 8, 16],
                        help='Number of goals')

    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'batch'],
                        help='Inference mode')

    parser.add_argument('--scenarios_file', type=str, default=None,
                        help='JSON file with scenarios for batch mode')

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    print("🚀 GBWM Inference System")
    print("=" * 50)
    print(f"📁 Model: {model_path.name}")
    print(f"🎯 Goals: {args.num_goals}")
    print(f"🔧 Mode: {args.mode}")
    print("=" * 50)

    try:
        # Initialize evaluator
        print("🔄 Loading trained model...")
        evaluator = GBWMEvaluator(str(model_path))
        print("✅ Model loaded successfully!")

        if args.mode == 'interactive':
            interactive_session(evaluator, args.num_goals)

        elif args.mode == 'batch':
            if args.scenarios_file:
                with open(args.scenarios_file, 'r') as f:
                    scenarios = json.load(f)
            else:
                # Default scenarios
                scenarios = [
                    {'age': 25, 'wealth': 100000},
                    {'age': 30, 'wealth': 200000},
                    {'age': 35, 'wealth': 150000},
                    {'age': 40, 'wealth': 500000},
                    {'age': 28, 'wealth': 80000},
                    {'age': 33, 'wealth': 300000},
                ]

            results = batch_inference(evaluator, scenarios)

            # Save results
            output_file = model_path.parent / "inference_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\n💾 Results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()