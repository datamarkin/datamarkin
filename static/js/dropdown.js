document.addEventListener('DOMContentLoaded', () => {
    // Get all dropdown elements
    const $dropdowns = Array.prototype.slice.call(document.querySelectorAll('.dropdown, .has-dropdown'), 0);

    // Add click event to each dropdown trigger
    $dropdowns.forEach(dropdown => {
        const trigger = dropdown.querySelector('.dropdown-trigger');

        if (trigger) {
            trigger.addEventListener('click', (event) => {
                event.stopPropagation(); // Prevent this click from being caught by the document listener

                // Close all other dropdowns first
                $dropdowns.forEach(otherDropdown => {
                    if (otherDropdown !== dropdown && otherDropdown.classList.contains('is-active')) {
                        otherDropdown.classList.remove('is-active');
                    }
                });

                // Toggle the current dropdown
                dropdown.classList.toggle('is-active');
            });
        }
    });

    // Add click event to the document to close dropdowns when clicking outside
    document.addEventListener('click', () => {
        $dropdowns.forEach(dropdown => {
            if (dropdown.classList.contains('is-active')) {
                dropdown.classList.remove('is-active');
            }
        });
    });

    // Prevent clicks within the dropdown content from closing the dropdown
    const $dropdownContent = document.querySelectorAll('.dropdown-content');
    $dropdownContent.forEach(content => {
        content.addEventListener('click', (event) => {
            event.stopPropagation(); // Stop click event from propagating to document
        });
    });
});