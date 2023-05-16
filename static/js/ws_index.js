var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
  $messages.mCustomScrollbar();
  setTimeout(function() {
    //fakeMessage();
  }, 100);
});

function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}

function setDate(){
  d = new Date()
  if (m != d.getMinutes()) {
    m = d.getMinutes();
    $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
  }
}

function insertMessage() {
  msg = $('.message-input').val();
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
  setDate();
  $('.message-input').val(null);
  updateScrollbar();
  interact(msg);
  setTimeout(function() {
    //fakeMessage();
  }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})

// Connect to the Socket.IO server
const socket = io.connect('http://' + document.domain + ':' + location.port);

let typingElement;

socket.on('connect', () => {
  console.log('Connected to the server');
});

socket.on('response', (response) => {
  // Message Received
  // remove loading message
  $('.message.loading').remove();

  // Add message to chatbox
  if (!typingElement) {
    typingElement = $('<div id="message-typing" class="message new"><figure class="avatar"><img src="/static/res/7.png" /></figure><div class="text"></div></div>').appendTo($('.mCSB_container'));
  }
  let responseText = response['text'];

  if (responseText === '<end>') {
    // End of message, prepare for next message
    typingElement.removeAttr('id');
    typingElement = null;
  } else {
    // Add the characters one by one
    typingElement.find('.text').text(typingElement.find('.text').text() + responseText);
    updateScrollbar();
  }
});




function interact(message) {
  // loading message
  $('<div class="message loading new"><figure class="avatar"><img src="/static/res/7.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));

  // Send the message to the server
  socket.emit('message', message);
}
