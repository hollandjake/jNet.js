class jNet {
	constructor(topology,activation){
		//Check there are enough
		if (topology.length < 2){
			console.error("Network not initialised: Insufficient number of layer");
			return;
		}

		this.iterations = 0;
		this.error = 0;
		this.averageError = 0;
		this.errorCap = 200;

		this.numLayers = topology.length;
		this.network = new Array(this.numLayers);

		switch (activation){
			case "sigmoid":
				this.activation = jNet.sigmoid;
				this.activationDerivative = jNet.dSigmoid;
				break;
			case "tanh":
				this.activation = jNet.tanh;
				this.activationDerivative = jNet.dtanh;
				break;
			default:
				this.activation = jNet.tanh;
				this.activationDerivative = jNet.dtanh;
				break;
		}

		//Initialise Network
		for (let l = 0;l<this.numLayers;l++){
			this.network[l] = new Array(topology[l]);
			let numOutputs = l+1 == this.numLayers ? 0 : topology[l+1];
			for (let n = 0;n<topology[l]+1;n++){ //+1 is for bias
				this.network[l][n] = new Neuron(numOutputs,n);
			}
			this.network[l][topology[l]].output = 1;
		}
	}

	static getBest(arr){
		if (arr.length === 0) {
			return -1;
		}
		let max = arr[0];
		let maxIndex = 0;
		for (let i = 1; i < arr.length; i++) {
			if (arr[i] > max) {
				maxIndex = i;
				max = arr[i];
			}
		}
		return maxIndex;
	}
	//softmax
	static getConfidence(arr){
		let a = arr.map(function(x){return Math.exp(x)});
		return 	a.reduce(function(a,b){return Math.max(a, b)}) /
						a.reduce(function(a,b){return a+b});
	}

	train(input,expected){

		let prediction = this.predict(input);

		//Calculate Overall error
		this.error = 0;

		for (let n = 0;n<prediction.output.length;n++){
			let delta = expected[n]-prediction.output[n];
			this.error += Math.pow(delta,2);
		}
		this.error = Math.sqrt(this.error/expected.length); //RMS

		this.averageError = (this.averageError*this.errorCap+this.error) / (this.errorCap + 1);

		//Calculate Output layer Gradients
		let outputLayer = this.network[this.numLayers-1];
		for (let n = 0;n<outputLayer.length-1;n++){
			outputLayer[n].calcOutputGradients(expected[n],this.activationDerivative);
		}
		//Calculate Hidden Gradients
		for (let l = this.numLayers-2;l>0;l--){
			let hiddenLayer = this.network[l];
			let nextLayer = this.network[l+1];
			for (let n = 0;n<hiddenLayer.length;n++){
				hiddenLayer[n].calcHiddenGradients(nextLayer,this.activationDerivative);
			}
		}
		//Update Connection Weights
		for (let l = this.numLayers-1;l>0;l--){
			let layer = this.network[l];
			let prevLayer = this.network[l-1];

			for (let n = 0;n<layer.length-1;n++){
				layer[n].updateInputWeights(prevLayer,this.eta,this.alpha);
			}
		}

		return prediction;
	}

	predict(input){
		if (input.length != this.network[0].length-1){
			console.error("Cannot Predict: Insufficient Inputs");
			return;
		}

		//Initialise Inputs
		for (let i = 0;i<input.length;i++){
			this.network[0][i].setOutput(input[i]);
		}

		//Feedforward
		for (let l = 1;l<this.numLayers;l++){
			let prevLayer = this.network[l-1];
			for (let j = 0;j<this.network[l].length - 1;j++){
				this.network[l][j].feedForward(prevLayer,this.activation);
			}
		}

		//Gather outputs
		let output = new Array(this.network[this.numLayers-1].length-1); //skip bias neuron
		for (let i = 0;i<output.length;i++){
			output[i] = this.network[this.numLayers-1][i].getOutput();
		}
		let prediction = jNet.getBest(output);
		return {
			"output": output,
			"prediction": prediction,
			"confidence": jNet.getConfidence(output)
		};
	}
}
class Neuron {
	constructor(numOutputs,index){
		this.output;
		this.index = index;
		this.gradient;
		this.outputWeights = new Array(numOutputs);
		for (let c = 0;c<numOutputs;c++){
			this.outputWeights[c] = new Connection();
		}
	}
	setOutput(val){
		this.output = val;
	}
	getOutput(){
		return this.output;
	}
	feedForward(prevLayer,activation){
		let sum = 0;
		for (let i = 0;i<prevLayer.length;i++){
			sum += prevLayer[i].output * prevLayer[i].outputWeights[this.index].weight;
		}
		this.output = activation(sum);
	}
	calcOutputGradients(target,activationDerivative){
		let delta = target - this.output;
		this.gradient = delta * activationDerivative(this.output);
	}
	calcHiddenGradients(nextLayer,activationDerivative){
		let dow = 0;
		//Sum the contributions I make to the error of the nodes we feed into
		for (let n = 0;n<nextLayer.length-1;n++){
			dow += this.outputWeights[n].weight * nextLayer[n].gradient;
		}

		this.gradient = dow * activationDerivative(this.output);
	}
	updateInputWeights(prevLayer,eta,alpha){
		for (let n = 0;n<prevLayer.length;n++){
			let neuron = prevLayer[n];
			let oldDeltaWeight = neuron.outputWeights[this.index].deltaWeight;
			let newDeltaWeight =
				//Individual input, magnified by the gradient and train rate
				eta
				* neuron.output
				* this.gradient
				//Add MOMENTUM
				+ alpha
				* oldDeltaWeight;
			neuron.outputWeights[this.index].deltaWeight = newDeltaWeight;
			neuron.outputWeights[this.index].weight += newDeltaWeight;
		}
	}
}
class Connection {
	constructor(){
		this.weight = Math.random()*2-1;
		this.deltaWeight = 0;
	}
}

//Activation Functions
jNet.sigmoid = function(x) {
	var y = 1 / (1 + Math.exp(-x));
	return y;
}
jNet.dSigmoid = function(x) {
	let i = jNet.sigmoid(x);
	return i * (1 - i);
}

jNet.tanh = function(x) {
	return Math.tanh(x);
}
jNet.dtanh = function(x) {
	return 1 - Math.pow(x,2);
}
