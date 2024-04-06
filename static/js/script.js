document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('upload-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', document.getElementById('file').files[0]);
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                throw new Error('Failed to receive response from server');
            }
            const data = await response.json();
            document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
        } catch (error) {
            console.error('Error:', error.message);
            // Display error message to the user
            document.getElementById('result').innerText = 'An error occurred. Please try again.';
        }
    });

    document.getElementById('file').addEventListener('change', function() {
        const fileInput = document.getElementById('file');
        const clearBtn = document.getElementById('clear-btn');
        const fileLabel = document.getElementById('file-label');

        if (fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
            clearBtn.style.display = 'inline-block';
        } else {
            fileLabel.textContent = 'Choose File';
            clearBtn.style.display = 'none';
        }
    });

    document.getElementById('clear-btn').addEventListener('click', function() {
        const fileInput = document.getElementById('file');
        fileInput.value = ''; 
        const fileLabel = document.getElementById('file-label');
        fileLabel.textContent = 'Choose File';
        this.style.display = 'none'; 
    });
});
