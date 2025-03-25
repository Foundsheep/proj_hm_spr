function updateFormAction(event) {
    event.preventDefault();

    // Get the selected radio value
    const selectedMethod = document.querySelector('input[name="method"]:checked').value;

    // Mapping for the selected method and api address
    const actionMapping = {
        "spr": "/gen/",
        "ssw": "/steel-spot-welding/",
    }

    // Get the form element
    const form = document.getElementById("methodForm");

    // Set the new address
    form.action = actionMapping[selectedMethod] || "/";

    // Submit the form
    form.submit();
}