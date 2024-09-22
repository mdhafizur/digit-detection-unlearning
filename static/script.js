
document
	.getElementById('predict-form')
	.addEventListener('submit', function (event) {
		event.preventDefault();
		var formData = new FormData(this);
		var spinner = document.getElementById('predict-spinner');
		spinner.style.display = 'inline-block';

		fetch('/predict', {
			method: 'POST',
			body: formData,
		})
			.then((response) => response.json())
			.then((data) => {
				document.getElementById('predict-result').textContent =
					'Predicted Digit: ' + data.digit;
				document.getElementById('predict-accuracy').textContent =
					'Accuracy: ' + (data.accuracy * 100).toFixed(2) + '%';
				spinner.style.display = 'none';
			})
			.catch((error) => {
				console.error('Error:', error);
				document.getElementById('predict-result').textContent =
					'Error: Unable to predict the digit.';
				spinner.style.display = 'none';
			});
	});

document
	.getElementById('unlearn-form')
	.addEventListener('submit', function (event) {
		event.preventDefault();

		// Get the selected digit and algorithm
		var digit = document.getElementById('digit').value;
		var algorithm = document.getElementById('algorithm').value;
		var spinner = document.getElementById('unlearn-spinner');
		spinner.style.display = 'inline-block';

		// Call visualization for selected algorithm
  		visualizeUnlearningProcess(algorithm);
		return
		// Determine the appropriate endpoint based on the selected algorithm
		let endpoint = '';
		switch (algorithm) {
			case 'sisa':
				endpoint = `/sisa_unlearn/${digit}`;
				break;
			case 'approx':
				endpoint = `/approx_unlearn/${digit}`;
				break;
			case 'certified':
				endpoint = `/certified_unlearn/${digit}`;
				break;
			case 'finetune':
				endpoint = `/finetune_unlearn/${digit}`;
				break;
			default:
				endpoint = `/unlearn/${digit}`;  // Default is standard unlearning
		}

		// Make the fetch request to the appropriate endpoint
		fetch(endpoint, {
			method: 'POST',
		})
			.then((response) => response.json())
			.then((data) => {
				// Display the result message
				document.getElementById('unlearn-result').textContent = data.message;
				spinner.style.display = 'none';
			})
			.catch((error) => {
				// Handle errors
				console.error('Error:', error);
				document.getElementById('unlearn-result').textContent =
					'Error: Unable to unlearn the digit.';
				spinner.style.display = 'none';
			});
	});


function appendLogMessage(message) {
	var logMessages = document.getElementById('log-messages');
	var logParagraph = document.createElement('p');
	logParagraph.textContent = message;
	logMessages.appendChild(logParagraph);
}

var socket = io.connect(
	window.location.protocol +
	'//' +
	document.domain +
	':' +
	window.location.port
);
console.log('socket', socket);

socket.on('log_message', function (message) {
	appendLogMessage(message);
});


function visualizeUnlearningProcess(algorithm) {
  const visualizationContainer = document.getElementById('algorithm-visualization');
  visualizationContainer.innerHTML = ''; // Clear previous content

  switch (algorithm) {
    case 'sisa':
      // SISA Visualization: Show Shards being retrained
      visualizationContainer.innerHTML = `
        <h4>Visualizing SISA Unlearning</h4>
        <p>SISA (Sharded, Isolated, and Aggregated) unlearning retrains only a subset of the dataset. Here’s how it works:</p>
        <ul>
          <li><strong>Step 1:</strong> The dataset is divided into <span id="shard-count">5</span> shards.</li>
          <li><strong>Step 2:</strong> Only the affected shard is retrained, while others remain unchanged.</li>
        </ul>
      `;
      const shardsText = document.createElement('div');
      shardsText.innerHTML = 'Shards Before Retraining:';
      visualizationContainer.appendChild(shardsText);

      // Create shards
      for (let i = 0; i < 5; i++) {
        const shard = document.createElement('div');
        shard.className = 'shard';
        shard.textContent = `Shard ${i + 1}`;
        shard.style.display = 'inline-block';
        shard.style.width = '60px';
        shard.style.height = '60px';
        shard.style.margin = '10px';
        shard.style.border = '1px solid #4caf50';
        shard.style.backgroundColor = '#4caf50';
        shard.style.transition = 'all 0.5s';
        visualizationContainer.appendChild(shard);
      }

      setTimeout(() => {
        const shards = document.getElementsByClassName('shard');
        shards[2].classList.add('retrained'); // Simulate retraining shard 3
        shards[2].style.backgroundColor = '#ff9800';
        shards[2].textContent = 'Retrained Shard 3';
        const resultText = document.createElement('p');
        resultText.innerHTML = `
          <strong>Result:</strong> Only <span style="color: #ff9800">Shard 3</span> was retrained. Other shards remain untouched, ensuring efficient retraining.
        `;
        visualizationContainer.appendChild(resultText);
      }, 1000);
      break;

    case 'approx':
      // Approximate Unlearning Visualization: Show weight perturbations
      visualizationContainer.innerHTML = `
        <h4>Visualizing Approximate Unlearning</h4>
        <p>Approximate unlearning works by modifying the model's parameters to reduce the influence of the data being unlearned. Here's how it works:</p>
        <ul>
          <li><strong>Step 1:</strong> Perturb (adjust) the weights of the model to reduce the contribution of the unlearned data.</li>
          <li><strong>Step 2:</strong> Monitor the change in model accuracy during the process.</li>
        </ul>
      `;
      const perturbationBar = document.createElement('div');
      perturbationBar.style.width = '100%';
      perturbationBar.style.height = '30px';
      perturbationBar.style.backgroundColor = '#4caf50';
      perturbationBar.style.transition = 'width 2s';
      visualizationContainer.appendChild(perturbationBar);

      const accuracyText = document.createElement('p');
      accuracyText.innerHTML = 'Initial Model Accuracy: 100%';
      visualizationContainer.appendChild(accuracyText);

      // Simulate weight perturbation
      setTimeout(() => {
        perturbationBar.style.width = '60%'; // Simulate reduction in accuracy
        perturbationBar.style.backgroundColor = '#ff9800';
        accuracyText.innerHTML = 'Updated Model Accuracy: 60% (after weight perturbation)';
        const resultText = document.createElement('p');
        resultText.innerHTML = `
          <strong>Result:</strong> The weights of the model have been adjusted, reducing the influence of the unlearned data.
        `;
        visualizationContainer.appendChild(resultText);
      }, 1000);
      break;

    case 'certified':
      // Certified Unlearning Visualization: Show influence reduction
      visualizationContainer.innerHTML = `
        <h4>Visualizing Certified Unlearning</h4>
        <p>Certified unlearning adjusts model parameters to systematically reduce the impact of the unlearned data. Here's how it works:</p>
        <ul>
          <li><strong>Step 1:</strong> Estimate the influence of the unlearned data on the model.</li>
          <li><strong>Step 2:</strong> Adjust model parameters to reduce this influence.</li>
        </ul>
      `;
      const influenceBars = [];
      for (let i = 0; i < 5; i++) {
        const bar = document.createElement('div');
        bar.style.width = '80%';
        bar.style.height = '20px';
        bar.style.backgroundColor = '#4caf50';
        bar.style.marginBottom = '5px';
        bar.style.transition = 'width 2s';
        influenceBars.push(bar);
        visualizationContainer.appendChild(bar);
      }

      const influenceText = document.createElement('p');
      influenceText.innerHTML = 'Initial Influence Level: High';
      visualizationContainer.appendChild(influenceText);

      // Simulate influence reduction
      setTimeout(() => {
        influenceBars.forEach((bar, idx) => {
          bar.style.width = `${80 - idx * 10}%`; // Reduce influence step-by-step
          bar.style.backgroundColor = '#ff9800';
        });
        influenceText.innerHTML = 'Updated Influence Level: Reduced';
        const resultText = document.createElement('p');
        resultText.innerHTML = `
          <strong>Result:</strong> The model has reduced the influence of the unlearned data significantly.
        `;
        visualizationContainer.appendChild(resultText);
      }, 1000);
      break;

    case 'finetune':
      // Fine-Tune Visualization: Show focused retraining
      visualizationContainer.innerHTML = `
        <h4>Visualizing Fine-Tune Unlearning</h4>
        <p>Fine-tuning retrains specific sections of the model to remove the influence of the unlearned data. Here's how it works:</p>
        <ul>
          <li><strong>Step 1:</strong> Retrain the sections of the model most affected by the unlearned data.</li>
          <li><strong>Step 2:</strong> Monitor model performance during and after the retraining process.</li>
        </ul>
      `;
      const retrainBar = document.createElement('div');
      retrainBar.style.width = '50%';
      retrainBar.style.height = '30px';
      retrainBar.style.backgroundColor = '#4caf50';
      retrainBar.style.transition = 'width 2s';
      visualizationContainer.appendChild(retrainBar);

      const finetuneText = document.createElement('p');
      finetuneText.innerHTML = 'Model Accuracy Before Fine-Tuning: 50%';
      visualizationContainer.appendChild(finetuneText);

      // Simulate retraining
      setTimeout(() => {
        retrainBar.style.width = '80%'; // Retraining shows an improvement
        retrainBar.style.backgroundColor = '#ff9800';
        finetuneText.innerHTML = 'Model Accuracy After Fine-Tuning: 80%';
        const resultText = document.createElement('p');
        resultText.innerHTML = `
          <strong>Result:</strong> The model has been fine-tuned to improve its performance after unlearning.
        `;
        visualizationContainer.appendChild(resultText);
      }, 1000);
      break;

    default:
      visualizationContainer.innerHTML = '<p>Select an algorithm to visualize how it works.</p>';
      break;
  }
}
