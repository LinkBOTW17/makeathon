# osapiens-challenge

## 🚀 osapiens Challenge: Detecting Deforestation from Space

What if you could be certain that the coffee you’re enjoying didn’t come at the cost of cutting down trees older than you?

Satellite constellations continuously capture the Earth using optical and radar sensors, enabling monitoring of forests at unprecedented scale. While satellite imagery provides global coverage, turning it into reliable insight is difficult. 

Satellite data is noisy, heterogeneous, and varies across sensors, regions, and environmental conditions. Thus, existing solutions that detect deforestation often suffer from limited accuracy and poor generalization.

As global deforestation continues, the EU Deforestation Regulation (EUDR) is driving demand for large-scale geospatial monitoring by requiring companies to verify that their supply chains are deforestation-free. 

Building systems that can turn these imperfect signals into reliable, automated, and globally consistent evidence remains an open problem and the focus of this challenge.

---

### 🧠 The Challenge

Build a machine learning system that identifies deforestation events using multimodal satellite data.

You will work with pixel-wise time series from radar and optical imagery, foundation model embeddings, and multiple weak label sources. 

The key challenge is to detect deforestation despite noisy data and conflicting supervision, while building a solution that generalizes across different geographic regions.

---

### 💡 Your Mission

Build an ML system that:

- Detects deforestation events at the pixel level using multimodal satellite time series
- Combines multimodal inputs such as Sentinel 1, Sentinel 2, and foundation model embeddings
- Generalizes to unseen geographic regions

---

### 🧰 What You’ll Have Access To

- Sentinel-1 radar and Sentinel-2 optical imagery as time series data
- Precomputed foundation model embeddings per pixel
- Multiple weak label sources for deforestation
- Ready to use dataloaders

You are not limited to the provided data and tools. We encourage you to be creative and build the solution you believe works best for this problem.

---

### 💥 What We’re Looking For

We evaluate solutions based on a combination of quantitative performance on a hidden test set and qualitative evaluation by a jury of experts. Key factors include:

- Model design, reasoning, and effective use of data
- Handling of noisy labels and imagery
- Generalization across geographical regions
- Quantitative performance and scalability of the solution
- Clarity, interpretability, and overall quality of presentation

Additionally, there is a live leaderboard during the hackathon that compares the performance of each team’s predictions on a hidden test set. Keep in mind that the ultimate ranking will be determined by a combination of quantitative performance and qualitative evaluation by the jury, so focus on building a strong overall solution rather than just optimizing for the leaderboard.

---

### 🌍 Why It Matters

Deforestation is a global challenge with significant environmental impact, often driven by the expansion of agricultural supply chains. Under the EU Deforestation Regulation (EUDR), companies must ensure that their supply chains are not linked to deforestation for certain commodities. This poses a major challenge for many industries, as they need to monitor vast and complex supply chains across multiple regions.

A single coffee bag might have hundreds of origins. Coffee beans are often sourced from multiple smallholder farms, collected by local cooperatives, processed in regional facilities, and then traded through several intermediaries before reaching roasters and retailers. At each stage, batches are mixed, split, and reassembled, making it increasingly difficult to trace the exact origin of every bean.

Reliable and scalable AI systems for detecting deforestation are essential to enable transparent monitoring of global supply chains and to protect ecosystems worldwide. By participating in this challenge, you have the opportunity to contribute to a solution that can help combat deforestation and promote sustainable practices globally.

---

### 🎯 Bonus (Optional)

- Predict *when* deforestation occurred (month or year)
- Design methods to handle label uncertainty or estimate confidence
- Build a simple visualization or monitoring tool