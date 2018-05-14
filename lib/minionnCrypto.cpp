/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#include "minionnCrypto.h"
#include "minionnMath.h"


Int PMAXBITS;
int SLOTS;
SEALContext *context;
PolyCRTBuilder *crtbuilder;
KeyGenerator *generator;

/**
Initializes important params. Should be called first.
*/
void init(int slot_count){
  EncryptionParameters parms;
  SLOTS = slot_count;
  parms.set_poly_modulus("1x^" + to_string(SLOTS) + " + 1");
  //parms.set_coeff_modulus(BigUInt("FFFFFFFFFFFFFFFFFFFFFFFFC0001"))
//  parms.set_coeff_modulus(ChooserEvaluator::default_parameter_options().at(SLOTS));
  //TODO: This might not be correct. Is below the 2.3 equivalent of the line above?
  parms.set_coeff_modulus(coeff_modulus_192(2*SLOTS));
  parms.set_plain_modulus(PMAX);
  PMAXBITS = parms.plain_modulus().bit_count();

  // Create context, CRT Builder, and generator
  context = new SEALContext(parms);
  generator = new KeyGenerator(*context);
  crtbuilder = new PolyCRTBuilder(*context);
}

/**
Generates a public and secret key and stores them in the given path
*/
void gen_keys(string str_pk, string str_sk)
{
  // Generate key and retrieve it from generator
  PublicKey pk = generator->public_key();
  SecretKey sk = generator->secret_key();

  // Store keys in file
  ofstream pk_file (str_pk, ofstream::binary);
  pk.save(pk_file);
  ofstream sk_file (str_sk, ofstream::binary);
  sk.save(sk_file);
}

/**
    Calculate dimensions of W shaped according to SLOT size
    n is the number of batches that we store w in
*/
int calculate_batches(int w_size){
  int l = w_size % SLOTS;
  int n = 0;
  if (l == 0)
    n = w_size / SLOTS;
  else
    n = w_size / SLOTS + 1;

  return n;
}

/**
  Takes an unsigned x and returns it as BigUInt that is adjusted to PMAX
*/
BigUInt adjust_to_PMAX(uInt x){
  /*
  if (x > PMAX_HALF)
    return BigUInt(PMAXBITS, x - PMAX);
  else
    return BigUInt(PMAXBITS, x);
    */
  if (x >= 0)
		return BigUInt(PMAXBITS, static_cast<uint64_t>(x));
	else
		return BigUInt(PMAXBITS, static_cast<uint64_t>(PMAX + x));
}

/**
    Encrypts a given vector w and returns it as a vector of python bytes
*/
vector<py::bytes> encrypt_w(std::vector<Int>* in_w, string pkp){
  // Read PK from file
  ifstream pks_file (pkp, ifstream::binary);
  PublicKey pks;
  pks.load(pks_file);

  Encryptor encryptor(*context, pks);

  int n = calculate_batches(in_w->size());

  // shape w into a n*SLOTS matrix and take entries from in_w adjusted with PMAX
  vector<uint64_t> tmp_W(n*SLOTS, 0);
  for (size_t i = 0; i < in_w->size(); i ++)
    if ((*in_w)[i] >= 0)
      tmp_W[i] = static_cast<uint64_t>((*in_w)[i]);
    else
      tmp_W[i] = static_cast<uint64_t>(PMAX+(*in_w)[i]);

  // encrypt w
  ostringstream ostream(stringstream::binary);
  vector<py::bytes> out_w;
  for (int i = 0; i < n; i ++)
	{
    //Divide W into n batches of size SLOTS and encode
    vector<uint64_t> batch(tmp_W.begin() + i*SLOTS, tmp_W.begin() + (i+1)*SLOTS);
    Plaintext encodedBatch(context->parms().poly_modulus().coeff_count(), 0);
    crtbuilder->compose(batch, encodedBatch);
    
    // Encrypt each batch separately
    Ciphertext encryptedBatch(context->parms());
    encryptor.encrypt(encodedBatch,encryptedBatch);
    encryptedBatch.save(ostream);

    // add to batch list
    out_w.push_back(py::bytes(ostream.str()));

    // clear ostream
    ostream.str("");
  }

  return out_w;
}

/**
  Decrypts a vector of bytes (given as string) and returns it into
  the out vector as Integer (uint64)
*/
void decrypt_w(vector<string>* w_in, string skp, vector<Int>* w_out){
  // Read secret key
  ifstream sks_file (skp,std::ifstream::binary);
  SecretKey sks;
  sks.load(sks_file);

  // create SEAL objects
  Decryptor decryptor(*context, sks);

  // reserve space for SLOTS*batches elements on U (upper bound as last batch might be emptier)
  w_out -> reserve(w_in->size() * SLOTS);

  // iterate through batches
  istringstream istream(stringstream::binary);
  for(size_t i = 0; i < w_in->size() ; i++){
      // read encrypted w
      istream.str("");
      istream.str(w_in->at(i));
      Ciphertext encryptedW(context->parms());
      encryptedW.load(istream);

      // decrypt w
      Plaintext encodedW(context->parms().poly_modulus().coeff_count(), 0);
      decryptor.decrypt(encryptedW, encodedW);
      //decompose it in batches to a BigUInt vector of size slot_count
      vector<uint64_t> tmp(SLOTS, 0);
      crtbuilder->decompose(encodedW, tmp);

      // iterate through tmp and push each element to U, adjusted for PMAX
      /*for (size_t j = 0; j < tmp.size(); j ++)
      {
          uint64_t t = (tmp[j].pointer())[0];
          w_out -> push_back(moduloPMAX(t));
      }*/

      for(uint64_t const& value: tmp){
        w_out -> push_back(moduloPMAX(value));
      }

  }

}

/**
  Performs the precomputation of the client.
  This is basically a computation of U = r*w - v for every given w
*/
vector<py::bytes> client_precomputation(vector<string>* w_in,
  vector<uInt>* r_in, vector<uInt>* v_in ){

  // create SEAL objects
  Evaluator evaluator(*context);

  // iterate through batches
  istringstream istream(stringstream::binary);
  ostringstream ostream(stringstream::binary);
  vector<py::bytes> out_w;

  // For every batch
  for(size_t i = 0; i < w_in->size() ; i++){
    // read encrypted w
    istream.str("");
    istream.str(w_in->at(i));
    Ciphertext encryptedW(context->parms());
    encryptedW.load(istream);

    // take r and v of this batch
    //  and put them into vectors of BigUint adjusted to PMAX
    vector<uint64_t> rr(SLOTS, 0);
    vector<uint64_t> vv(SLOTS, 0);
    
    // Check that r (and as such v) has enough size remaining for a whole batch
    size_t r_size = (size_t) SLOTS;
    if(r_in->size() - i * SLOTS < SLOTS){
      r_size = (size_t) (r_in->size() - i * SLOTS);
    }

    rr.reserve(r_size);
    vv.reserve(r_size);

    // Now iterate over r (or remaining r) and put it into rr and vv
    //  If not a whole batch can be filled, it remains 0 (as defined in rr and vv)
    for (int j = 0; j < r_size; j ++){
      rr[j] = r_in -> at(SLOTS*i + j);
      vv[j] = v_in -> at(SLOTS*i + j);
    }
    
    Plaintext encodedR(context->parms().poly_modulus().coeff_count(), 1);
    crtbuilder->compose(rr, encodedR);
    Plaintext encodedV(context->parms().poly_modulus().coeff_count(), 0);
    crtbuilder->compose(vv, encodedV);

    // calculate U = r*w - v
    evaluator.multiply_plain(encryptedW, encodedR);
    evaluator.sub_plain(encryptedW, encodedV);

    // save U back to the list
    encryptedW.save(ostream);
    out_w.push_back(py::bytes(ostream.str()));

    // clear ostream
    ostream.str("");
  }

  return out_w;

}
