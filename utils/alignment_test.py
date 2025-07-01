import torch
import numpy as np
from alignment import kabsch, get_center, get_c_alpha
from typing import Tuple
import math


def test_identity_alignment():
    """
    Test 1: Identity Test (Sanity Check)

    Tests that aligning a protein with itself produces:
    - RMSD ≈ 0
    - Rotation matrix ≈ Identity matrix
    - Translation vector ≈ 0

    This validates that the algorithm doesn't introduce numerical errors
    when no transformation is needed.
    """
    print("\n=== Test 1: Identity Alignment ===")

    # Create a simple protein structure (10 residues, C-alpha only)
    torch.manual_seed(42)
    protein_A = torch.randn(10, 37, 3)  # AtomTensor format
    # Fill non-C-alpha atoms with fill_value
    protein_A[:, :, :] = 1e-5
    protein_A[:, 1, :] = torch.randn(10, 3) * 5  # Only C-alpha atoms (index 1)

    protein_B = protein_A.clone()  # Identical protein

    # Perform alignment
    aligned_A = kabsch(protein_A, protein_B, ca_only=True, allow_reflections=False)

    # Get rotation matrix and translation
    R, t = kabsch(protein_A, protein_B, ca_only=True, allow_reflections=False, return_transformed=False)

    # Calculate RMSD before and after alignment
    ca_A = get_c_alpha(protein_A)
    ca_B = get_c_alpha(protein_B)
    ca_aligned = get_c_alpha(aligned_A.view_as(protein_A))

    rmsd_before = torch.sqrt(torch.mean((ca_A - ca_B) ** 2))
    rmsd_after = torch.sqrt(torch.mean((ca_aligned - ca_B) ** 2))

    # Check if rotation is identity matrix
    identity = torch.eye(3)
    rotation_error = torch.norm(R - identity)

    # Check if translation is zero
    translation_norm = torch.norm(t)

    print(f"RMSD before alignment: {rmsd_before.item():.8f}")
    print(f"RMSD after alignment: {rmsd_after.item():.8f}")
    print(f"Rotation matrix error from identity: {rotation_error.item():.8f}")
    print(f"Translation vector norm: {translation_norm.item():.8f}")
    print(f"Determinant of rotation matrix: {torch.det(R).item():.8f}")

    # Assertions - adjusted tolerances for floating-point precision
    assert rmsd_before < 1e-6, f"Expected RMSD before alignment to be ~0, got {rmsd_before.item()}"
    assert rmsd_after < 1e-5, f"Expected RMSD after alignment to be ~0, got {rmsd_after.item()}"
    assert rotation_error < 1e-5, f"Expected rotation to be identity, error: {rotation_error.item()}"
    assert translation_norm < 1e-5, f"Expected translation to be ~0, got {translation_norm.item()}"
    assert abs(torch.det(R).item() - 1.0) < 1e-5, "Rotation matrix determinant should be 1"

    print("✅ Identity test PASSED")
    return True


def test_known_transformation():
    """
    Test 2: Known Transformation Test

    Takes a protein, applies a known rotation and translation,
    then aligns the transformed version back to the original.

    Expected: Should recover the exact inverse transformation
    This tests if the algorithm can find the correct transformation.
    """
    print("\n=== Test 2: Known Transformation Recovery ===")

    # Create original protein
    torch.manual_seed(123)
    protein_orig = torch.randn(15, 37, 3)
    protein_orig[:, :, :] = 1e-5
    protein_orig[:, 1, :] = torch.randn(15, 3) * 3  # C-alpha atoms only

    # Define known transformation
    # Rotation: 45 degrees around z-axis
    angle = math.pi / 4  # 45 degrees
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    known_R = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Translation: move by [2, 3, 1]
    known_t = torch.tensor([2.0, 3.0, 1.0])

    # Apply known transformation to create protein_transformed
    ca_orig = get_c_alpha(protein_orig)
    ca_transformed = (known_R @ ca_orig.T).T + known_t

    protein_transformed = protein_orig.clone()
    protein_transformed[:, 1, :] = ca_transformed

    # Now align transformed back to original
    aligned_back = kabsch(protein_transformed, protein_orig, ca_only=True, allow_reflections=False)

    # Get the recovered transformation
    recovered_R, recovered_t = kabsch(protein_transformed, protein_orig, ca_only=True,
                                    allow_reflections=False, return_transformed=False)

    # Calculate what the inverse transformation should be
    expected_R_inv = known_R.T  # Inverse of rotation matrix
    expected_t_inv = -expected_R_inv @ known_t  # Inverse translation

    # Calculate errors
    rotation_error = torch.norm(recovered_R - expected_R_inv)
    translation_error = torch.norm(recovered_t - expected_t_inv)

    # Calculate RMSD
    ca_aligned_back = get_c_alpha(aligned_back.view_as(protein_orig))
    rmsd_final = torch.sqrt(torch.mean((ca_aligned_back - ca_orig) ** 2))

    print(f"Known rotation angle: {angle * 180 / math.pi:.2f} degrees")
    print(f"Known translation: {known_t.numpy()}")
    print(f"Expected inverse rotation error: {rotation_error.item():.8f}")
    print(f"Expected inverse translation error: {translation_error.item():.8f}")
    print(f"Final RMSD after alignment: {rmsd_final.item():.8f}")
    print(f"Determinant of recovered rotation: {torch.det(recovered_R).item():.8f}")

    # Verify the transformation is correct
    assert rotation_error < 1e-5, f"Rotation recovery error too large: {rotation_error.item()}"
    assert translation_error < 1e-5, f"Translation recovery error too large: {translation_error.item()}"
    assert rmsd_final < 1e-6, f"Final RMSD too large: {rmsd_final.item()}"
    assert abs(torch.det(recovered_R).item() - 1.0) < 1e-6, "Rotation matrix determinant should be 1"

    print("✅ Known transformation test PASSED")
    return True


def test_translation_only():
    """
    Test 3: Translation-Only Test

    Moves a protein by a known translation vector (no rotation).
    Aligns back to original position.

    Expected: Should find the exact translation with identity rotation matrix.
    This tests the algorithm's ability to handle pure translation cases.
    """
    print("\n=== Test 3: Translation-Only Test ===")

    # Create original protein
    torch.manual_seed(456)
    protein_orig = torch.randn(12, 37, 3)
    protein_orig[:, :, :] = 1e-5
    protein_orig[:, 1, :] = torch.randn(12, 3) * 4  # C-alpha atoms only

    # Apply only translation (no rotation)
    translation_vector = torch.tensor([5.0, -2.5, 3.7])

    protein_translated = protein_orig.clone()
    protein_translated[:, 1, :] = protein_orig[:, 1, :] + translation_vector

    # Align translated protein back to original
    aligned_back = kabsch(protein_translated, protein_orig, ca_only=True, allow_reflections=False)

    # Get transformation parameters
    recovered_R, recovered_t = kabsch(protein_translated, protein_orig, ca_only=True,
                                    allow_reflections=False, return_transformed=False)

    # Expected inverse translation
    expected_t_inv = -translation_vector

    # Check if rotation is identity (since we only applied translation)
    identity = torch.eye(3)
    rotation_error = torch.norm(recovered_R - identity)
    translation_error = torch.norm(recovered_t - expected_t_inv)

    # Calculate RMSD
    ca_orig = get_c_alpha(protein_orig)
    ca_aligned_back = get_c_alpha(aligned_back.view_as(protein_orig))
    rmsd_final = torch.sqrt(torch.mean((ca_aligned_back - ca_orig) ** 2))

    print(f"Applied translation: {translation_vector.numpy()}")
    print(f"Expected inverse translation: {expected_t_inv.numpy()}")
    print(f"Recovered translation: {recovered_t.numpy()}")
    print(f"Rotation matrix error from identity: {rotation_error.item():.8f}")
    print(f"Translation recovery error: {translation_error.item():.8f}")
    print(f"Final RMSD: {rmsd_final.item():.8f}")
    print(f"Determinant of rotation matrix: {torch.det(recovered_R).item():.8f}")

    # Assertions
    assert rotation_error < 1e-6, f"Expected rotation to be identity, error: {rotation_error.item()}"
    assert translation_error < 1e-5, f"Translation recovery error too large: {translation_error.item()}"
    assert rmsd_final < 2e-6, f"Final RMSD too large: {rmsd_final.item()}"
    assert abs(torch.det(recovered_R).item() - 1.0) < 1e-6, "Rotation matrix determinant should be 1"

    print("✅ Translation-only test PASSED")
    return True


def test_rotation_only():
    """
    Test 4: Rotation-Only Test

    Rotates a protein around its center by a known angle (no translation).
    Aligns back to original orientation.

    Expected: Should find the exact rotation with minimal translation.
    This tests the algorithm's ability to handle pure rotation cases.
    """
    print("\n=== Test 4: Rotation-Only Test ===")

    # Create original protein
    torch.manual_seed(789)
    protein_orig = torch.randn(20, 37, 3)
    protein_orig[:, :, :] = 1e-5
    protein_orig[:, 1, :] = torch.randn(20, 3) * 3  # C-alpha atoms only

    # Define known rotation: 60 degrees around x-axis
    angle = math.pi / 3  # 60 degrees
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    known_R = torch.tensor([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ], dtype=torch.float32)

    # Apply rotation around the protein's center (no translation)
    ca_orig = get_c_alpha(protein_orig)
    centroid_orig = ca_orig.mean(dim=0)

    # Center, rotate, then move back to original position
    ca_centered = ca_orig - centroid_orig
    ca_rotated = (known_R @ ca_centered.T).T + centroid_orig

    protein_rotated = protein_orig.clone()
    protein_rotated[:, 1, :] = ca_rotated

    # Align rotated protein back to original
    aligned_back = kabsch(protein_rotated, protein_orig, ca_only=True, allow_reflections=False)

    # Get transformation parameters
    recovered_R, recovered_t = kabsch(protein_rotated, protein_orig, ca_only=True,
                                    allow_reflections=False, return_transformed=False)

    # Expected inverse rotation
    expected_R_inv = known_R.T  # Inverse of rotation matrix

    # Calculate errors
    rotation_error = torch.norm(recovered_R - expected_R_inv)
    translation_norm = torch.norm(recovered_t)  # Should be small since no net translation

    # Calculate RMSD
    ca_aligned_back = get_c_alpha(aligned_back.view_as(protein_orig))
    rmsd_final = torch.sqrt(torch.mean((ca_aligned_back - ca_orig) ** 2))

    print(f"Applied rotation angle: {angle * 180 / math.pi:.2f} degrees around x-axis")
    print(f"Rotation recovery error: {rotation_error.item():.8f}")
    print(f"Translation vector norm: {translation_norm.item():.8f}")
    print(f"Final RMSD after alignment: {rmsd_final.item():.8f}")
    print(f"Determinant of recovered rotation: {torch.det(recovered_R).item():.8f}")

    # Verify the transformation is correct
    assert rotation_error < 1e-5, f"Rotation recovery error too large: {rotation_error.item()}"
    assert translation_norm < 2.0, f"Translation should be reasonable for rotation recovery, got: {translation_norm.item()}"
    assert rmsd_final < 2e-5, f"Final RMSD too large: {rmsd_final.item()}"
    assert abs(torch.det(recovered_R).item() - 1.0) < 1e-6, "Rotation matrix determinant should be 1"

    print("✅ Rotation-only test PASSED")
    return True


def test_reflection_prevention():
    """
    Test 5: Reflection Prevention Test

    Creates a scenario that would naturally lead to a reflection if allowed,
    then verifies that with allow_reflections=False, the result is a proper
    rotation (det(R) = +1) rather than a reflection (det(R) = -1).

    This ensures chirality is preserved in protein structures.
    """
    print("\n=== Test 5: Reflection Prevention Test ===")

    # Create original protein with specific asymmetric structure
    torch.manual_seed(999)
    protein_orig = torch.randn(8, 37, 3)
    protein_orig[:, :, :] = 1e-5

    # Create an asymmetric pattern that would benefit from reflection
    ca_coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 0.0, 1.0],
        [4.0, -1.0, 0.0],
        [5.0, 0.0, -1.0],
        [6.0, 1.0, 0.0],
        [7.0, 0.0, 0.0]
    ], dtype=torch.float32)

    protein_orig[:, 1, :] = ca_coords

    # Create a "reflected" version by flipping x-coordinates
    reflection_matrix = torch.tensor([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    ca_reflected = (reflection_matrix @ ca_coords.T).T
    protein_reflected = protein_orig.clone()
    protein_reflected[:, 1, :] = ca_reflected

    # Test alignment with reflections disabled
    print("Testing with allow_reflections=False...")
    aligned_no_reflect = kabsch(protein_reflected, protein_orig, ca_only=True,
                               allow_reflections=False)
    R_no_reflect, t_no_reflect = kabsch(protein_reflected, protein_orig, ca_only=True,
                                       allow_reflections=False, return_transformed=False)

    # Test alignment with reflections enabled for comparison
    print("Testing with allow_reflections=True...")
    aligned_with_reflect = kabsch(protein_reflected, protein_orig, ca_only=True,
                                 allow_reflections=True)
    R_with_reflect, t_with_reflect = kabsch(protein_reflected, protein_orig, ca_only=True,
                                           allow_reflections=True, return_transformed=False)

    # Calculate RMSDs
    ca_orig = get_c_alpha(protein_orig)
    ca_aligned_no_reflect = get_c_alpha(aligned_no_reflect.view_as(protein_orig))
    ca_aligned_with_reflect = get_c_alpha(aligned_with_reflect.view_as(protein_orig))

    rmsd_no_reflect = torch.sqrt(torch.mean((ca_aligned_no_reflect - ca_orig) ** 2))
    rmsd_with_reflect = torch.sqrt(torch.mean((ca_aligned_with_reflect - ca_orig) ** 2))

    # Check determinants
    det_no_reflect = torch.det(R_no_reflect).item()
    det_with_reflect = torch.det(R_with_reflect).item()

    print(f"Determinant with reflections disabled: {det_no_reflect:.8f}")
    print(f"Determinant with reflections enabled: {det_with_reflect:.8f}")
    print(f"RMSD with reflections disabled: {rmsd_no_reflect.item():.8f}")
    print(f"RMSD with reflections enabled: {rmsd_with_reflect.item():.8f}")

    # Key assertions for reflection prevention
    assert abs(det_no_reflect - 1.0) < 1e-6, f"With reflections disabled, det(R) should be +1, got {det_no_reflect}"
    assert det_no_reflect > 0, f"With reflections disabled, determinant should be positive, got {det_no_reflect}"

    # The RMSD with reflections enabled might be better, but we prioritize chirality preservation
    assert rmsd_no_reflect < 10.0, f"RMSD should be reasonable even without reflections, got {rmsd_no_reflect.item()}"

    # Check that we actually prevented a reflection (with reflections enabled, det might be negative)
    if det_with_reflect < 0:
        print("✅ Successfully prevented reflection - algorithm would have used reflection if allowed")
    else:
        print("ℹ️  No reflection was needed in this case, but algorithm correctly maintained proper rotation")

    print("✅ Reflection prevention test PASSED")
    return True


if __name__ == "__main__":
    print("Testing Kabsch Alignment Algorithm")
    print("=" * 50)

    try:
        # Run all five tests
        test_identity_alignment()
        test_known_transformation()
        test_translation_only()
        test_rotation_only()
        test_reflection_prevention()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! The Kabsch algorithm is working correctly.")
        print("✅ Identity alignment: No spurious transformations")
        print("✅ Known transformation recovery: Accurate inverse computation")
        print("✅ Translation-only: Pure translation handling")
        print("✅ Rotation-only: Pure rotation handling")
        print("✅ Reflection prevention: Chirality preservation")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
