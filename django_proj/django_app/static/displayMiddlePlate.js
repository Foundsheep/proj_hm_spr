// Function to toggle visibility of middle_type label
function toggleMiddleTypeVisibility() {
  var plateCount = document.querySelector('input[name="plate_count"]:checked').value;
  var middleFieldset = document.getElementById("middle_fieldset");
  // var middleTypeLabel = document.getElementById('middle_type_label');
  // var middleType = document.getElementById('middle_type');
  // var middleThicknessLabel = document.getElementById("middle_thickness_label");
  // var middleThicknessInput = document.getElementById("middle_thickness_input");

  // Hide the middle_type label if plate count is 2, show it otherwise
  if (plateCount === '2') {
    middleFieldset.style.display = "none";
    // middleTypeLabel.style.display = 'none';
    // middleType.style.display = "none";
    // middleThicknessLabel.style.display = "none";
  } else {
    middleFieldset.style.display = "block";
    // middleTypeLabel.style.display = 'inline';
    // middleType.style.display = 'inline';
    // middleThicknessLabel.style.display = "inline";
  }
}

// Run the toggle function when the page loads and when plate count changes
window.addEventListener('DOMContentLoaded', toggleMiddleTypeVisibility);
document.querySelectorAll('input[name="plate_count"]').forEach(function(radio) {
radio.addEventListener('change', toggleMiddleTypeVisibility);
});