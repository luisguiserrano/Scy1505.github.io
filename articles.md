---
layout: default
title: Posts
---

<div id="articles">
  <h1 class="pageTitle">General</h1>
  <ul class="posts noList">
    {% for post in site.categories.general %}
      <li>
      	<span class="date">{{ post.date | date_to_string }}</span>
      	<h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
      	<p class="description">{% if post.description %}{{ post.description  | strip_html | strip_newlines | truncate: 120 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 120 }}{% endif %}</p>
      </li>
    {% endfor %}
  </ul>
  <h1 class="pageTitle">Reinforcement</h1>
  <ul class="posts noList">
    {% for post in site.categories.reinforcement %}
      <li>
        <span class="date">{{ post.date | date_to_string }}</span>
        <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
        <p class="description">{% if post.description %}{{ post.description  | strip_html | strip_newlines | truncate: 120 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 120 }}{% endif %}</p>
      </li>
    {% endfor %}
  </ul>

  <h1 class="pageTitle">Neural Networks for Image Reccognition</h1>
  <ul class="posts noList">
    {% for post in site.categories.NN_for_image_reccognition %}
      <li>
        <span class="date">{{ post.date | date_to_string }}</span>
        <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
        <p class="description">{% if post.description %}{{ post.description  | strip_html | strip_newlines | truncate: 120 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 120 }}{% endif %}</p>
      </li>
    {% endfor %}
  </ul>


  <h1 class="pageTitle">Math and Stats</h1>
  <ul class="posts noList">
    {% for post in site.categories.math %}
      <li>
        <span class="date">{{ post.date | date_to_string }}</span>
        <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
        <p class="description">{% if post.description %}{{ post.description  | strip_html | strip_newlines | truncate: 120 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 120 }}{% endif %}</p>
      </li>
    {% endfor %}
  </ul>



  
</div>