const grid = document.getElementById("grid");
const submitButton = document.getElementById("submit");
const refreshButton = document.getElementById("refresh");
let isDrawing = false;

function createGrid() {
    for (let i = 0; i < 28 * 28; i++) {
        const pixel = document.createElement("div");
        pixel.classList.add("pixel");
        pixel.addEventListener("mousedown", () => {
            isDrawing = true;
            pixel.classList.add("active");
        });
        pixel.addEventListener("mouseenter", () => {
            if (isDrawing) {
                pixel.classList.add("active");
            }
        });
        grid.appendChild(pixel);
    }
}

function getMatrix() {
    const pixels = document.querySelectorAll(".pixel");
    let matrix = [];
    let row = [];
    
    pixels.forEach((pixel, index) => {
        const isActive = pixel.classList.contains("active") ? 1 : 0;
        row.push(isActive);

        // If we've reached the end of a row (28 pixels), push the row to the matrix and start a new one
        if ((index + 1) % 28 === 0) {
            matrix.push(row);
            row = [];
        }
    });
    
    return matrix;
}


submitButton.addEventListener("click", async () => {
    const matrix = getMatrix();
    console.log(JSON.stringify(matrix));

    try {
        // Sending the matrix data to a Flask server
        const response = await fetch('http://127.0.0.1:5000/api/matrix', {
            method: 'POST', // or 'PUT'
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(matrix), // body data type must match "Content-Type" header
        });

        const jsonResponse = await response.json(); // parses JSON response into native JavaScript objects
        console.log(jsonResponse);
    } catch (error) {
        console.error('Error:', error);
    }
});


refreshButton.addEventListener("click", () => {
    const pixels = document.querySelectorAll(".pixel");
    pixels.forEach(pixel => {
        pixel.classList.remove("active");
    });
});

// Stop drawing when the mouse is released
document.addEventListener("mouseup", () => {
    isDrawing = false;
});

createGrid();
