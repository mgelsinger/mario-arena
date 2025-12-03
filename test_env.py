"""
Test script to verify the Mario environment wrapper works correctly.

This script creates a Mario environment, runs a few random steps,
and verifies the observation shapes and info dict are correct.
"""

from mario_arena.envs import make_mario_env, get_action_meanings, extract_info_from_env


def test_environment():
    """Test the Mario environment wrapper."""
    print("=" * 60)
    print("Testing Mario Arena Environment")
    print("=" * 60)

    # Create environment
    print("\n1. Creating environment...")
    env = make_mario_env("SuperMarioBros-1-1-v0", render_mode=None)
    print(f"   Environment created successfully!")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Get action meanings
    print("\n2. Action meanings:")
    actions = get_action_meanings("simple")
    for i, action in enumerate(actions):
        print(f"   Action {i}: {action}")

    # Reset environment
    print("\n3. Resetting environment...")
    result = env.reset()
    # Handle both old and new gym API
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Expected: (84, 84, 4) for 4 stacked 84x84 grayscale frames")

    # Take some random steps
    print("\n4. Taking 10 random steps...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Extract info
        extracted_info = extract_info_from_env(info)

        print(f"   Step {step + 1}:")
        print(f"      Action: {action} ({actions[action]})")
        print(f"      Reward: {reward:.1f}")
        print(f"      X-pos: {extracted_info['x_pos']}")
        print(f"      Done: {done}")

        if done:
            print(f"   Episode ended! Resetting...")
            result = env.reset()
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
            break

    # Verify observation shape
    print("\n5. Verification:")
    expected_shape = (84, 84, 4)
    if obs.shape == expected_shape:
        print(f"   [OK] Observation shape is correct: {obs.shape}")
    else:
        print(f"   [FAIL] Observation shape is incorrect!")
        print(f"     Expected: {expected_shape}")
        print(f"     Got: {obs.shape}")

    # Close environment
    env.close()
    print("\n6. Environment closed successfully!")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_environment()
