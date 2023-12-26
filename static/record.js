document.getElementById("recordButton").onclick = function() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            const audioChunks = [];
            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });
            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks);
                sendAudioToServer(audioBlob);
            });

            setTimeout(() => {
                mediaRecorder.stop();
            }, 10000); // Stops recording after 10 seconds
        });
};

// ... [rest of your JavaScript code for recording]

function sendAudioToServer(blob) {
    const formData = new FormData();
    formData.append("audio", blob, "audio.mp3");

    fetch("/upload", {
        method: "POST",
        body: formData
    }).then(response => response.json()).then(data => {
        displayList(data); // Call a function to display the list
    });
}

function displayList(data) {
    const listContainer = document.createElement('ul');
    data.forEach(item => {
        const listItem = document.createElement('li');
        listItem.textContent = item;
        listContainer.appendChild(listItem);
    });
    document.body.appendChild(listContainer);
}

