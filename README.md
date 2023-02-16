# 🤖 Machine Learning Experiments

This is a repository to keep all of my Machine Learning experiments in one place. Especially when working through tutorials, the project can feel easy to discard yet often contains useful scaffolding or re-usable functions for later projects. 

> Feel free to clone, fork, or even contribute if you wish!

## ✍🏾️ Embedding & Vector Search

### 🔍Twitter Jobs Vector Search (with OpenAI Embeddings)

I wanted to understand more about embeddings, and I also wanted to find interesting job listings with a little less manual work involved. So, I thought I'd take a shot at using the OpenAI Embeddings API to help me find the jobs I'd be interested in.

I [wrote about the project here](https://ciaran.co.za/read/finding-jobs-with-openai-embeddings), and it turned out to be a pretty quick and simple idea to bring to life. 

### 📚 EPUB Vector Search

Experimenting with getting embeddings and performing vector search in a local environment (ie. not using the OpenAI API), I created a notebook to read all EPUBs from my [Calibre Library](https://calibre-ebook.com/) and allow me to search through them for different phrases. 

It might take a while to create indices for all the pages in your library if you're not using a GPU (or if you have a really large library), but the semantic search works really well - especially when run on textbooks or instruction manuals.

## 🖌️ GAN Tutorial Series

This repository contains a collection of PyTorch programming lessons, based on the tutorials by [Aladdin Persson on YouTube](https://www.youtube.com/c/AladdinPersson). The lessons cover a wide range of topics, from basic PyTorch operations to advanced techniques for building neural networks.

Not every exercise produced outstanding results. However, the aim was more to learn about GANs and how they work in order to improve my comprehension of arXiv papers. Machine learning is a field that interests me a lot, so while I'm no expert, I would like to be able to keep up with more than just big developments, and formulate my own opinions of various implementations and their practical applications.

### Examples

With CycleGAN, for exampe, I ran it on a dataset of real-life images, and Ukiyo-e style Japanese art. The exercise produced results such as the following:

|Style|Input|Output|
|---|---|---|
|Photograph|![Photograph Input](real_in.png)|![Ukiyo-e Style Output](art-out.png)|
|Ukiyo-e|![Ukiyo-e Input](art_in.png)|![Photographic Style Output](real-out.png)|

### Completed Exercises

| Example | Video |
| --- | --- |
| **Simple GAN** | [Link](https://www.youtube.com/watch?v=OljTVUVzPpM) |
| **DCGAN (Deep Convolutional GAN)** | [Link](https://www.youtube.com/watch?v=IZtv9s_Wx9I) |
| **WGAN (Wasserstein GAN)** | [Link](https://www.youtube.com/watch?v=pG0QZ7OddX4) |
| **Conditional GAN** | [Link](https://www.youtube.com/watch?v=Hp-jWm2SzR8) |
| **Pix2Pix** | [Link](https://www.youtube.com/watch?v=SuddDSqGRzg) |
| **CycleGAN** | [Link](https://www.youtube.com/watch?v=4LktBHGCNfw) |
