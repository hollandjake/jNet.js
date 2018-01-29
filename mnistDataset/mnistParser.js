class MNIST{
	constructor(trainingExamples,testExamples,padding,finishedLoadingFunction){
		this.imageSize = 28;
		this.imagePadding = padding;

		this.trainingData = [];
		this.trainingExamples = trainingExamples;
		this.trainingCounter = 0;
		this.trainingEpochs = 0;

		this.testData = [];
		this.testExamples = testExamples;
		this.testCounter = 0;
		this.testEpochs = 0;

		//Create bufferCanvas
		this.bufferCanvas = document.createElement("canvas");
		this.bufferCanvas.width = this.imageSize;
		this.bufferCanvas.height = this.imageSize;
		this.bufferCanvas.style.display = 'none';
		//CANVAS.parentNode.appendChild(this.bufferCanvas);

		this.bufferCtx = this.bufferCanvas.getContext("2d");
		this.bufferCtx.imageSmoothingEnabled = false;

		this.tempCanvas = document.createElement("canvas");
		this.tempCanvas.width = this.imageSize;
		this.tempCanvas.height = this.imageSize;
		this.tempCanvas.style.display = 'none';

		this.tempCtx = this.tempCanvas.getContext("2d");
		this.tempCtx.imageSmoothingEnabled = false;

		//Grab datasets
		let self = this;
		self.getBinary("https://rawgit.com/hollandjake/jNet.js/master/mnistDataset/train-labels",function(data){
			let labels = data;
			if (labels.length < trainingExamples){
				console.error("Not enough entries in the training datset");
			}
			let trainingData = new Array(labels.length);
			self.getBinary("https://rawgit.com/hollandjake/jNet.js/master/mnistDataset/train-images",function(data){
				for (let i = 0;i<labels.length;i++){
					trainingData[i] = {
						"idNumeric": labels[i],
						"id": MNIST.arrayFromIndex(labels[i],10),
						"data": data.subarray(i*784,(i+1)*784)
					};
				}
				self.trainingData = MNIST.shuffle(trainingData).slice(0,trainingExamples);

				self.getBinary("https://rawgit.com/hollandjake/jNet.js/master/mnistDataset/test-labels",function(data){
					let labels = data;
					if (labels.length < testExamples){
						console.error("Not enough entries in the test datset");
					}
					let testData = new Array(labels.length);
					self.getBinary("https://rawgit.com/hollandjake/jNet.js/master/mnistDataset/test-images",function(data){
						for (let i = 0;i<labels.length;i++){
							testData[i] = {
								"idNumeric": labels[i],
								"id": MNIST.arrayFromIndex(labels[i],10),
								"data": data.subarray(i*784,(i+1)*784)
							};
						}
						self.testData = MNIST.shuffle(testData).slice(0,testExamples);
						finishedLoadingFunction();
					},16);
				},8);
			},16);
		},8);
	}

	static shuffle(array){
		var currentIndex = array.length;

		// While there remain elements to shuffle...
		while (0 !== currentIndex) {

			// Pick a remaining element...
			let randomIndex = Math.floor(Math.random() * currentIndex);
			currentIndex -= 1;

			// And swap it with the current element.
			let temporaryValue = array[currentIndex];
			array[currentIndex] = array[randomIndex];
			array[randomIndex] = temporaryValue;
		}

		return array;
	}
	static arrayFromIndex(index,range){
		let array = new Array(range).fill(0);
		array[index] = 1;
		return array;
	}
	getBinary(filename,callback,offset){
		var oReq = new XMLHttpRequest();
		oReq.open("GET", filename, true);
		oReq.responseType = "arraybuffer";

		oReq.onload = function (oEvent) {
			var arrayBuffer = oReq.response; // Note: not oReq.responseText
			if (arrayBuffer) {
				callback(new Uint8Array(arrayBuffer,offset));
			}
		};
		oReq.send(null);
	}
	canvasToData(canvas){
		let dataStream = {
			"idNumeric": null,
			"id": null,
			"data": []
		}
		let ctx = this.bufferCtx;
		ctx.clearRect(0,0,this.imageSize,this.imageSize);
		ctx.drawImage(canvas,0,0,this.imageSize,this.imageSize);
		let imageData = ctx.getImageData(0,0,this.imageSize,this.imageSize).data;
		for (let i = 3;i<imageData.length;i+=4){
			dataStream.data.push(imageData[i]);
		}
		dataStream.image = ctx.canvas;
		return this.preprocess(dataStream,this.imageSize,this.imagePadding);
	}
	getNext(type,withImage){
		let dataset;
		let counter;
		if (type=="training"){
			this.trainingCounter++;
			if (this.trainingCounter >= this.trainingExamples) {
				this.trainingCounter = 1;
				this.trainingEpochs++;
			}
			counter = this.trainingCounter;
			dataset = this.trainingData;
		} else if (type=="test"){
			this.testCounter++;
			if (this.testCounter >= this.testExamples) {
				this.testCounter = 1;
				this.testEpochs++;
			}
			counter = this.testCounter;
			dataset = this.testData;
		}
		if(!dataset[counter-1].image){
			dataset[counter-1] = this.preprocess(dataset[counter - 1],this.imageSize,this.imagePadding);
		}
		return dataset[counter - 1];
	}
	preprocess(dataStream,targetSize,padding){
		//Render an Image representation and add to dataStream

		//Bounding box
		let boundingBox = {
			top: {x:this.imageSize,y:this.imageSize},
			bottom: {x:0,y:0},
		}

		let tCtx = this.tempCtx;
		tCtx.clearRect(0,0,this.tempCanvas.width,this.tempCanvas.height);
		this.bufferCtx.clearRect(0,0,this.bufferCanvas.width,this.bufferCanvas.height);
		let imageData = tCtx.createImageData(targetSize,targetSize);
		let data = imageData.data;
		for (let i = 0;i<dataStream.data.length;i++){
			data[i*4] = 255;//red
			data[i*4+1] = 255;//green
			data[i*4+2] = 255;//blue
			data[i*4+3] = dataStream.data[i];//alpha

			//Bounding Box Part 1 -Find Corners
			if (dataStream.data[i] > 0) {
				let x = i % targetSize;
				let y = Math.floor(i / targetSize);

				if (boundingBox.top.x > x){boundingBox.top.x = x;}
				if (boundingBox.top.y > y){boundingBox.top.y = y;}
				if (boundingBox.bottom.x < x){boundingBox.bottom.x = x;}
				if (boundingBox.bottom.y < y){boundingBox.bottom.y = y;}
			}
		}
		//Apply Padding
		boundingBox.top.x-=padding;
		boundingBox.top.y-=padding;
		boundingBox.bottom.x+=padding;
		boundingBox.bottom.y+=padding;

		tCtx.putImageData(imageData,0,0);

		this.bufferCtx.drawImage(this.tempCanvas,boundingBox.top.x,boundingBox.top.y,boundingBox.bottom.x-boundingBox.top.x,boundingBox.bottom.y-boundingBox.top.y,0,0,targetSize,targetSize);
		let tCanvas = document.createElement("canvas").getContext("2d");
		tCanvas.canvas.width = this.bufferCanvas.width;
		tCanvas.canvas.height = this.bufferCanvas.height;
		tCanvas.drawImage(this.bufferCanvas,0,0);
		dataStream.image = tCanvas.canvas;

		dataStream.data = [];
		imageData = this.bufferCtx.getImageData(0,0,targetSize,targetSize).data;
		for (let i = 3;i<imageData.length;i+=4){
			dataStream.data.push(imageData[i]/255);
		}
		return dataStream;
	}

}
