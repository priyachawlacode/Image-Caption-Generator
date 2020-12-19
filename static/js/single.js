window.onload = function(){
  let params = new URLSearchParams(location.search);
  var temp=params.get('type')
  console.log(temp);
fetch('https://goodquotesapi.herokuapp.com/tag/'+temp)
  .then(function (response) {
    return response.json();
  })
  .then(function (data) {
    console.log(data)
    appendData(data,temp);
  })
  .catch(function (err) {
    console.log(err);
  });
}

function tocopy(id) {
    var text = document.getElementById(id).innerText;
    var elem = document.createElement("textarea");
    document.body.appendChild(elem);
    elem.value = text;
    elem.select();
    document.execCommand("copy");
    document.body.removeChild(elem);
}

function appendData(data,type) {
  
  var mainContainer = document.getElementById("bread");
  var mainContainer1 = document.getElementById("myData1");
  var mainContainer2 = document.getElementById("myData2");
  var stop_loader= document.getElementsByClassName("load");
  stop_loader[0].style.display="none";

  var div = document.createElement("div");
  div.innerHTML = `<nav aria-label="breadcrumb" style="background-color:#20c997">
                      <ol class="breadcrumb">
                        <li class="breadcrumb-item">Quotes</li>
                        <li class="breadcrumb-item active" aria-current="page">${type}</li>
                      </ol>
                    </nav>`;
  mainContainer.appendChild(div)

  for (var i = 0; i < data.quotes.length; i=i+2) {

    var div1 = document.createElement("div");
    var div2 = document.createElement("div");
   
    div1.innerHTML = `<div class="col mb-4 text-justified" data-aos="fade">
                        <button type="button" class="btn-clipboard btn-dark" onclick="tocopy(${i})">Copy</button>                        
                        <div class="card" style="background-color:black;margin-right:2%">
                          <div class="card-body quote">
                          <p class="card-title" id=${i}>${data.quotes[i].quote}</p>
                          <h5 class="card-title text-right">~ ${data.quotes[i].author}</h5>
                        </div>
                      </div>
                    </div>`;
    div2.innerHTML = `<div class="col mb-4" data-aos="fade">
                        <button type="button" class="btn-clipboard btn-dark" onclick="tocopy(${i})">Copy</button>                        
                        <div class="card" style="background-color:black;margin-right:2%">
                          <div class="card-body quote">
                          <p class="card-title" id=${i}>${data.quotes[i+1].quote}</p>
                          <h5 class="card-title text-right">~ ${data.quotes[i+1].author}</h5>
                        </div>
                      </div>
                    </div>`;

    mainContainer1.appendChild(div1)
    mainContainer2.appendChild(div2)
    
  }
}
