
var canvas, ctx;
var mouseX,
	mouseY,
	mouseDown = 0;
var touchX, touchY;
var lastX, lastY;

function init() {
	canvas = document.getElementById('canvas');
	ctx = canvas.getContext('2d');
	ctx.fillStyle = 'black';
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	if (ctx) {
		canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
		canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
		window.addEventListener('mouseup', sketchpad_mouseUp, false);
		canvas.addEventListener('touchstart', sketchpad_touchStart, false);
		canvas.addEventListener('touchmove', sketchpad_touchMove, false);
	}
}

function draw(ctx, x, y, size, isDown) {
	if (isDown) {
		ctx.beginPath();
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 15;
		ctx.lineJoin = ctx.lineCap = 'round';
		ctx.moveTo(lastX, lastY);
		ctx.lineTo(x, y);
		ctx.closePath();
		ctx.stroke();
	}
	lastX = x;
	lastY = y;
}

function sketchpad_mouseDown() {
	getMousePos();
	mouseDown = 1;
	draw(ctx, mouseX, mouseY, 12, false);
}

function sketchpad_mouseUp() {
	mouseDown = 0;
}

function sketchpad_mouseMove(e) {
	getMousePos(e);
	if (mouseDown == 1) {
		draw(ctx, mouseX, mouseY, 12, true);
	}
}

function getMousePos(e) {
	if (!e) var e = event;
	if (e.offsetX) {
		mouseX = e.offsetX;
		mouseY = e.offsetY;
	} else if (e.layerX) {
		mouseX = e.layerX;
		mouseY = e.layerY;
	}
}

function sketchpad_touchStart() {
	getTouchPos();
	draw(ctx, touchX, touchY, 12, false);
	event.preventDefault();
}

function sketchpad_touchMove(e) {
	getTouchPos(e);
	draw(ctx, touchX, touchY, 12, true);
	event.preventDefault();
}

function getTouchPos(e) {
	if (!e) var e = event;
	if (e.touches) {
		if (e.touches.length == 1) {
			var touch = e.touches[0];
			touchX = touch.pageX - touch.target.offsetLeft;
			touchY = touch.pageY - touch.target.offsetTop;
		}
	}
}

document
	.getElementById('clear-canvas')
	.addEventListener('click', function () {
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctx.fillStyle = 'black';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
	});

document
	.getElementById('predict-canvas')
	.addEventListener('click', function () {
		const spinner = document.getElementById('canvas-predict-spinner');
		spinner.style.display = 'inline-block';

		canvas.toBlob((blob) => {
			const formData = new FormData();
			formData.append('image', blob, 'digit.png');

			fetch('/predict', {
				method: 'POST',
				body: formData,
			})
				.then((response) => response.json())
				.then((data) => {
					document.getElementById('canvas-predict-result').textContent =
						'Predicted Digit: ' + data.digit;
					document.getElementById('canvas-predict-accuracy').textContent =
						'Accuracy: ' + (data.accuracy * 100).toFixed(2) + '%';
					spinner.style.display = 'none';
				})
				.catch((error) => {
					console.error('Error:', error);
					document.getElementById('canvas-predict-result').textContent =
						'Error: Unable to predict the digit.';
					spinner.style.display = 'none';
				});
		}, 'image/png');
	});

document.getElementById('image').addEventListener('change', function () {
	var reader = new FileReader();
	reader.onload = function (e) {
		document.getElementById('image-preview').src = e.target.result;
		document.getElementById('image-preview').style.display = 'block';
	};
	reader.readAsDataURL(this.files[0]);
});