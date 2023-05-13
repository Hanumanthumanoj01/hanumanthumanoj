(function () {
    [...document.querySelectorAll(".control")].forEach(button => {
        button.addEventListener("click", function() {
            document.querySelector(".active-btn").classList.remove("active-btn");
            this.classList.add("active-btn");
            document.querySelector(".active").classList.remove("active");
            document.getElementById(button.dataset.id).classList.add("active");
        })
    });
    document.querySelector(".theme-btn").addEventListener("click", () => {
        document.body.classList.toggle("light-mode");
    })
})();

function sendEmail(e) {
    e.preventDefault()
    var name = document.getElementById("name").value
    var email = document.getElementById("email").value
    var subject = document.getElementById("subject").value
    var message = document.getElementById("message").value
    if(!name || !email || !subject || !message) {
        alert("Fill all fields and try again")
        return
    }
	Email.send({
        isNotMendrill: true,
        Username: 'hanumanthumanoj27@gmail.com',
        Password: 'Manoj@123',
        Host: 'smtp.elasticemail.com',
        Port: '2525',
        From: "hanumanthumanoj27@gmail.com",
        To : 'hmanoj.ece@gmail.com',
        Subject :subject,
        Body :  `email: ${email} <br/> name:${name} <br/> Subject: ${subject} <br/> message: ${message}`,
	}).then(res =>{ 
        alert('Message has been send');
        document.getElementById("name").value = ""
        document.getElementById("email").value = ""
        document.getElementById("subject").value = ""
        document.getElementById("message").value = ""
    }, function(error) {
        console.log('FAILED...', error);
    });

}